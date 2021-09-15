#!/bin/bash

# Office-31
python main.py  --data_path_source ./data/datasets/Office31/  --src amazon  --data_path_target_tr ./data/datasets/Office31/  --tar_tr webcam  --data_path_target_te ./data/datasets/Office31/  --tar_te webcam  --log ./checkpoints/office31  --vda #


python main.py  --data_path_source ./data/datasets/Office31/  --src dslr  --data_path_target_tr ./data/datasets/Office31/  --tar_tr webcam  --data_path_target_te ./data/datasets/Office31/  --tar_te webcam  --log ./checkpoints/office31  --vda #

               
python main.py  --data_path_source ./data/datasets/Office31/  --src webcam  --data_path_target_tr ./data/datasets/Office31/  --tar_tr dslr  --data_path_target_te ./data/datasets/Office31/  --tar_te dslr  --log ./checkpoints/office31  --vda #


python main.py  --data_path_source ./data/datasets/Office31/  --src amazon  --data_path_target_tr ./data/datasets/Office31/  --tar_tr dslr  --data_path_target_te ./data/datasets/Office31/  --tar_te dslr  --log ./checkpoints/office31  --vda #


python main.py  --data_path_source ./data/datasets/Office31/  --src dslr  --data_path_target_tr ./data/datasets/Office31/  --tar_tr amazon  --data_path_target_te ./data/datasets/Office31/  --tar_te amazon  --log ./checkpoints/office31  --vda #


python main.py  --data_path_source ./data/datasets/Office31/  --src webcam  --data_path_target_tr ./data/datasets/Office31/  --tar_tr amazon  --data_path_target_te ./data/datasets/Office31/  --tar_te amazon  --log ./checkpoints/office31  --vda #


