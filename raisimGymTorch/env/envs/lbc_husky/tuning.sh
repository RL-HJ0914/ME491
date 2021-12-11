#!/bin/bash

controldt_coeffs="0.07"
vel_coeffs="66 69 72"
near_coeffs="95 100 105 110 115"
cfg_khj="$(cat cfg_khj.yaml)"
for controldt_coeff in $controldt_coeffs
do for vel_coeff in $vel_coeffs
do for near_coeff in $near_coeffs
do
  echo "$cfg_khj" > cfg.yaml
  sed -i "s/controldt_coeff/$controldt_coeff/g" cfg.yaml
  sed -i "s/vel_coeff/$vel_coeff/g" cfg.yaml
  sed -i "s/near_coeff/$near_coeff/g" cfg.yaml
  echo "control_dt,vel,near:  $controldt_coeff $vel_coeff $near_coeff" >> tuning_log.txt
  python runner.py
done
done
done
