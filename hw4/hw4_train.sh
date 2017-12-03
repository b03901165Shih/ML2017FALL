#!/bin/sh

# Uncomment the following two lines for training word embedding model
# But the model will not be the same due to random factor (affect testing)

#wget 'https://www.dropbox.com/s/2ymhiorzdsre18w/testing_data.txt?dl=1' -O testing_data.txt
#python hw4_embed.py $1 $2

wget 'https://www.dropbox.com/s/o46z0ap3dz2xuvn/model_embed.bin?dl=1' -O model_embed.bin
#wget 'https://www.dropbox.com/s/kyw3f66vhocas2x/RNN_model_frame_forSelf.json?dl=1' -O RNN_model_frame_forSelf.json
#wget 'https://www.dropbox.com/s/m1egc6halahn958/RNN_model_weight_forSelf.h5?dl=1' -O RNN_model_weight_forSelf.h5
python hw4_RNN_train.py $1 $2

#repeat multiple times of finetune for better accuracy
python hw4_RNN_finetune.py $1 $2
