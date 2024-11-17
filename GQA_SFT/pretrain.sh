#!/bin/bash

folder="/root/project/My_projects/Transformer/GQA_SFT/data/datas"
files=($(ls "$folder"))
num_files=${#files[@]}

for ((j=0; j<10; j++)); do
  for ((i=0; i<$num_files; i++)); do
    fn="${files[$i]}"

    if [ "$i" -gt -1 ]; then
      deepspeed train.py --data_file "$folder/$fn" --ss $i
    fi
  done
done
