from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lbc_husky
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse


# task specification
task_name = "husky_navigation"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('-c', '--changelist', help='changelist of this learning', type=str, default='')

args = parser.parse_args()
mode = args.mode
weight_path = args.weight
changelist=args.changelist

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."


# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(lbc_husky.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_iteration_steps = n_steps * env.num_envs
total_steps = 0

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         device) # Can change initial Policy std
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp", task_path + "/runner.py", task_path+"/../../../algo/ppo/ppo.py"])
# tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

# save changelist
f= open(saver.data_dir+"/changelist.txt",'w')
f.write(changelist)
f.close()

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=cfg['runner']['num_learning_epochs'],# default 4
              gamma=cfg['runner']['gamma'], #default 0.996
              lam=0.95,
              num_mini_batches=cfg['runner']['num_mini_batches'],
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              )


scheduler = torch.optim.lr_scheduler.MultiStepLR(ppo.optimizer, milestones=[], gamma=0.3)


if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

T_threshold = 6  ;
T_interval = 0.08 ;

for update in range(1000000):
    start = time.time()
    env.reset()
    reward_sum = 0
    done_sum = 0
    completed_sum = 0
    average_dones = 0.

    if update % cfg['environment']['eval_every_n'] == 0 or (update > 2000 and update % 30 == 0):
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        data_tags = env.get_step_data_tag()
        data_size = 0
        data_mean = np.zeros(shape=(len(data_tags), 1), dtype=np.float32)
        data_square_sum = np.zeros(shape=(len(data_tags), 1), dtype=np.float32)
        data_min = np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.float32)
        data_max = -np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.float32)

        for step in range(n_steps):
            frame_start = time.time()
            obs = env.observe(False)
            action = loaded_graph.architecture(torch.from_numpy(obs).cpu())
            reward, dones, completed = env.step(action.cpu().detach().numpy())
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)
            data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)

        data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)

        data_std = np.sqrt((data_square_sum - data_size * data_mean * data_mean) / (data_size - 1 + 1e-16))

        for data_id in range(len(data_tags)):
            ppo.writer.add_scalar(data_tags[data_id]+'/mean', data_mean[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id]+'/std', data_std[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id]+'/min', data_min[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id]+'/max', data_max[data_id], global_step=update)

        env.stop_video_recording()
        env.turn_off_visualization()

        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    # actual training
    for step in range(n_steps):
        obs = env.observe()
        action = ppo.observe(obs)
        reward, dones, not_completed = env.step(action)
        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + sum(dones)
        reward_sum = reward_sum + sum(reward)
        completed_sum = completed_sum + sum(not_completed)

    # data constraints - DO NOT CHANGE THIS BLOCK
    total_steps += env.num_envs * n_steps
    if total_steps > 20000000:
        break

    # take st step to get value obs
    obs = env.observe()
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_performance = reward_sum / total_iteration_steps
    average_dones = done_sum / total_iteration_steps
    avg_rewards.append(average_performance)

    # curriculum update. Implement it in Environment.hpp
    env.curriculum_callback()

    end = time.time()

    scheduler.step()

    if update % 10 == 0:
        ppo.writer.add_scalar('avg completion time', completed_sum / env.num_envs * cfg['environment']['control_dt'], global_step=update)

    if completed_sum / env.num_envs * cfg['environment']['control_dt'] <T_threshold:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        data_tags = env.get_step_data_tag()
        data_size = 0
        data_mean = np.zeros(shape=(len(data_tags), 1), dtype=np.float32)
        data_square_sum = np.zeros(shape=(len(data_tags), 1), dtype=np.float32)
        data_min = np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.float32)
        data_max = -np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.float32)

        for step in range(n_steps):
            frame_start = time.time()
            obs = env.observe(False)
            action = loaded_graph.architecture(torch.from_numpy(obs).cpu())
            reward, dones, completed = env.step(action.cpu().detach().numpy())
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)
            data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)

        data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)

        data_std = np.sqrt((data_square_sum - data_size * data_mean * data_mean) / (data_size - 1 + 1e-16))

        for data_id in range(len(data_tags)):
            ppo.writer.add_scalar(data_tags[data_id]+'/mean', data_mean[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id]+'/std', data_std[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id]+'/min', data_min[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id]+'/max', data_max[data_id], global_step=update)

        env.stop_video_recording()
        env.turn_off_visualization()

        env.reset()
        env.save_scaling(saver.data_dir, str(update))
        T_threshold-=T_interval

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("avg reward: ", '{:0.10f}'.format(average_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("avg completion time: ", '{:0.6f}'.format(completed_sum / env.num_envs * cfg['environment']['control_dt'])))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_iteration_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_iteration_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')
