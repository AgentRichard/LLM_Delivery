#!/bin/bash

folder="datas/SFT_preprocess/sft_data"
files=($(ls "$folder"))
num_files=${#files[@]}

for ((j=0; i<6; j++)); do
  for ((i=0; i<$num_files; i++)); do
    fn="${files[$i]}"

    if [ "$i" -gt -1 ]; then
      deepspeed sft_pretrain.py --data_file "$folder/$fn" --ss $i
    fi
  done
done
