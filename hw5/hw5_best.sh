#!/bin/sh

# training
# python hw5_MF_train_NN.py <trainging data path> <user.csv path>
# fine-Tuning
# python hw5_MF_finetune_NN.py <training data path> <user.csv path>

# testing
python hw5_MF_test_NN.py $1 $4 $2 
