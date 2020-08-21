#!/bin/bash

folder=$1
minfile=min_$1.log
sota=200.0
for file in `\find ./checkpoints/${folder} -name '*.pth'| sed 's!^.*/!!'`; do
    python predict_wnet_noskip.py --input sar.npy --no-save --model checkpoints/${folder}/${file}
    # python predict_wnet.py --input sar.npy --no-save --model checkpoints/${folder}/${file}

    min=`cat ${minfile}`
    if [ `echo "$sota > $min" | bc` == 1 ]; then
        sota=${min}
        echo ${min},${file} > $1.log
    fi
done
