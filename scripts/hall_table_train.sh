export PYTHONPATH=./:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 \
  python train_ace.py -c "./cfg/hall_table_train.yaml"
