#!/bin/sh

wget 'https://www.dropbox.com/s/bmd4gmk4l5r4onn/hw3_model_self.h5?dl=1' -O hw3_model_self.h5
python hw3_cnn_test.py $1 $2
