#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=0 \
#     python test_ace.py  "${dataset_path}/7scenes_chess" "${model_path}/7scenes_chess.pt"

CUDA_VISIBLE_DEVICES=0 \
    python test_ace.py -c "./cfg/test_7scene_chess_v2.yaml"
