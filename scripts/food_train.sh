export PYTHONPATH=./:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 \
  python train_ace.py -c "./cfg/food_train.yaml"
