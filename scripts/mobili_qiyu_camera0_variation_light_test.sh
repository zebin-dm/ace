#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 \
    python test_ace.py -c "./cfg/mobili_qiyu_camera0_variation_light_test.yaml"
