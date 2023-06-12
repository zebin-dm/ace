#!/usr/bin/env bash

model_path="/mnt/nas/share-all/caizebin/03.dataset/ace/pretrained/ace_models/7Scenes"
dataset_path="/mnt/nas/share-all/caizebin/03.dataset/ace/7scenes_ace"
CUDA_VISIBLE_DEVICES=0 \
    python test_ace.py  "${dataset_path}/7scenes_chess" "${model_path}/7scenes_chess.pt"
