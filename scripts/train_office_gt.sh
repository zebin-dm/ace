export PYTHONPATH=./:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 \
  python train_ace.py -c "./cfg/train_office_gt.yaml"
