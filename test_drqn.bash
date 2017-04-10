#!/bin/bash

python -u drqn_atari.py --env Breakout-v0 --output ./output --input_shape 84 84 --replay_buffer_size 1000 --num_burn_in 250 --model_name lstm2 --num_train 2000 --target_reset_interval 500 --do_render T --eval_episodes 2 --num_frame 1 --online_train_interval 16 --timesteps 4 --save_interval 500

