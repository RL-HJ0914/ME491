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
import subprocess
import compete_script


task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."
variable_path= "env-200/4.96s-992iter/"
weight_dir = home_path + "/raisimGymTorch/data/husky_navigation/"+variable_path

file_list=os.listdir(weight_dir)
file_list_pt=[file for file in file_list if file.endswith(".pt")]
iter_nums= []
for file in file_list_pt:
    iter_nums.append(file.split('_', 1)[1].rsplit('.', 1)[0])

record=[] #record (iter, time)
best_time=8.0

iter_nums2=copy.deepcopy(iter_nums)

# 짤짤이 삭제
for iteration_number in iter_nums:
    if not (int(iteration_number) == 984 or int(iteration_number) ==992):
        iter_nums2.remove(iteration_number)

# main for문
for iteration_number in iter_nums2:
    # complete_time=subprocess.call("compete_script.py "+"-w "+variable_path+"full_"+str(iteration_number)+".pt")
    complete_time=compete_script.get_time(variable_path+"full_"+str(iteration_number)+".pt")
    best_time= min(best_time, complete_time)
    record.append((iteration_number, complete_time))

print("best time: ", best_time)
print("record: ", record)