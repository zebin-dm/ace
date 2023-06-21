#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 \
    python test_ace.py -c "./cfg/office_table_test.yaml"
