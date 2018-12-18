#!/bin/bash

data_dir=.

#train=$data_dir/arithmétique_train.csv
train=$data_dir/sinusoide_train.csv

#dev=$data_dir/arithmétique_dev.csv
dev=$data_dir/sinusoide_dev.csv

#test=$data_dir/arithmétique_test.csv
test=$data_dir/sinusoide_test.csv

input=4,5,6,7,8
output=9

model_file=$data_dir/my_model.json
epochs=500
verbose=0

python train_neural_network.py --epochs $epochs \
                               --save-model $model_file \
                               --train-input $train:$input \
                               --train-output $train:$output \
                               --dev-input $dev:$input \
                               --dev-output $dev:$output \
                               --test-input $test:$input \
                               --test-output $test:$output \
                               --optimization "adam" \
                               --recurrent
