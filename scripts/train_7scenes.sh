#!/usr/bin/env bash

# Find the path to the root of the repo.
# SCRIPT_PATH=$(dirname $(realpath -s "$0"))
# REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

# scenes=("7scenes_chess" "7scenes_fire" "7scenes_heads" "7scenes_office" "7scenes_pumpkin" "7scenes_redkitchen" "7scenes_stairs")

# training_exe="${REPO_PATH}/train_ace.py"
# testing_exe="${REPO_PATH}/test_ace.py"

# datasets_folder="${REPO_PATH}/datasets"
# out_dir="${REPO_PATH}/output/7Scenes"
# mkdir -p "$out_dir"



# for scene in ${scenes[*]}; do
#   python $training_exe "$datasets_folder/$scene" "$out_dir/$scene.pt"
#   python $testing_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" 2>&1 | tee "$out_dir/log_${scene}.txt"
# done

# for scene in ${scenes[*]}; do
#   echo "${scene}: $(cat "${out_dir}/log_${scene}.txt" | tail -5 | head -1)"
# done

scene="7scenes_chess"
datasets_folder="/mnt/nas/share-all/caizebin/03.dataset/ace/7scenes_ace"
out_dir="./output/7Scenes"
CUDA_VISIBLE_DEVICES=0 \
  python train_ace.py "$datasets_folder/$scene" "$out_dir/$scene.pt" -c "./cfg/train_7scene_chess.yaml"
# python test_ace.py "$datasets_folder/$scene" "$out_dir/$scene.pt" 2>&1 | tee "$out_dir/log_${scene}.txt"
# echo "${scene}: $(cat "${out_dir}/log_${scene}.txt" | tail -5 | head -1)"