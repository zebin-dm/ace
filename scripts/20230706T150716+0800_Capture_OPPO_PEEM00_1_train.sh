export PYTHONPATH=./:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 \
  python train_ace.py -c "./cfg/20230706T150716+0800_Capture_OPPO_PEEM00_1_train.yaml"
