#!/usr/bin/env bash

if [ ! -d "log" ]; then
    mkdir "log"
fi

output="log/yolo2-800x800-distin-split_sqrt-final=resume38-lr=0.001.log"
nohup python train.py 1>${output} 2>&1 &
