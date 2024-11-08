#!/bin/bash

folder="datas/data"
files=($(ls "$folder"))
num_files=${#files[@]}

for ((j=0; i<1; j++)); do
  for ((i=0; i<$num_files; i++)); do
    fn="${files[$i]}"

    if [ "$i" -gt -1 ]; then
      deepspeed pretrain.py --data_file "$folder/$fn" --ss $i
    fi
  done
done
