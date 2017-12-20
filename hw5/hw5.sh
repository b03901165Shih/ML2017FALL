#!/bin/sh

# training
# python hw5_MF_train.py <trainging data path>
# fine-Tuning
# python hw5_MF_finetune.py <training data path>

# testing
python hw5_MF_test.py $1 $2
