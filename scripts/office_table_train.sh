export PYTHONPATH=./:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 \
  python train_ace.py -c "./cfg/office_table_train.yaml"
