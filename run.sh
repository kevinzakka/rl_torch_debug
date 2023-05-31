#!/bin/bash

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 python train.py \
    --warmstart-steps 10000 \
    --max-steps 1000000 \
    --discount 0.99 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 \
    --tqdm-bar
