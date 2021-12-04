from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lbc_husky
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import time
import torch
import argparse
import copy
import numpy as np
import random
# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."
weight_dir = home_path + "/raisimGymTorch/data/husky_navigation/2021-12-01-15-11-25/"
file_list=os.listdir(weight_dir)
file_list_pt=[file for file in file_list if file.endswith(".pt")]
iter_nums= []
for file in file_list_pt:
    iter_nums.append(file.split('_', 1)[1].rsplit('.', 1)[0])


# config
cfg = YAML().load(open(weight_dir + "cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 200
env = VecEnv(lbc_husky.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

record=[] #record (iter, time)
best_time=8.0

iter_nums2=copy.deepcopy(iter_nums)

# for iteration_number in iter_nums:
#     if not (int(iteration_number) == 960 or int(iteration_number) == 968):
#         iter_nums2.remove(iteration_number)

for iteration_number in iter_nums2:
    env.seed(0)
    weight_path=weight_dir+"full_"+str(iteration_number)+".pt"
    print("Loaded weight from {}\n".format(weight_path))
    env.reset()
    completion_sum = 0
    done_sum = 0
    average_dones = 0.

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    max_steps = 80 ## 8 secs
    completed_sum = 0

    for step in range(max_steps):
        obs = env.observe(False)
        if (step==0):
            print(obs)
        action = loaded_graph.architecture(torch.from_numpy(obs).cpu())
        reward, dones, not_completed = env.step(action.cpu().detach().numpy())
        completed_sum = completed_sum + sum(not_completed)

    best_time= min(best_time, completed_sum / env.num_envs * cfg['environment']['control_dt'] )
    record.append((iteration_number, completed_sum / env.num_envs * cfg['environment']['control_dt']))
    print('----------------------------------------------------')
    print('{:<40} {:>6}'.format("avg completion time: ", '{:0.6f}'.format(completed_sum / env.num_envs * cfg['environment']['control_dt'])))
    print('----------------------------------------------------\n')

    env.turn_off_visualization()



print("best time: ", best_time)
print("record: ", record)