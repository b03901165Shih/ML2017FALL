#!/bin/sh

wget 'https://www.dropbox.com/s/o46z0ap3dz2xuvn/model_embed.bin?dl=1' -O model_embed.bin
#wget 'https://www.dropbox.com/s/j0f3zs6b1e94i13/RNN_model_frame_final.json?dl=1' -O RNN_model_frame_final.json
#wget 'https://www.dropbox.com/s/5h2eb3a2uw21201/RNN_model_weight_final.h5?dl=1' -O RNN_model_weight_final.h5

python hw4_RNN_test.py $1 $2
