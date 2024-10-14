#!/usr/bin/env bash

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)



org_wav_dir=$1
data_dir=data

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <wav_dir>"
    exit 1
fi

set -euo pipefail

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}
[ -e ${text} ] && rm ${text}

# prepare main dataset
echo "Prepare main dataset"
for spk in male female; do
    spk_wav_dir=${org_wav_dir}/${spk}
    # make scp, utt2spk, and spk2utt
    find $spk_wav_dir -follow -name "*.wav" | sort | while read -r filename; do
        id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
        echo "${spk}_${id} ${filename}" >> ${scp}
        echo "${spk}_${id} ${spk}" >> ${utt2spk}
    done
    metadata="${spk_wav_dir}/metadata.csv"
    sed -e "s/|/ /g" "${metadata}" | sed "s/\.wav//g"| sed -e "s/^/${spk}_/g" > ${text}  
done
# Splitting train, valid and test splits
echo "Train, valid and test split"
train_dir=${data_dir}/train
valid_dir=${data_dir}/valid
test_dir=${data_dir}/test

if [! -d $train_dir]; then
    mkdir -p $train_dir
fi

if [! -d $valid_dir]; then
    mkdir -p $valid_dir
fi

if [! -d $test_dir]; then
    mkdir -p $test_dir
fi
python local/split_set.py \
        --i $data_dir \
        --train_dir $train_dir \
        --valid_dir $valid_dir \
        --test_dir $test_dir

utils/utt2spk_to_spk2utt.pl ${train_dir}/utt2spk > ${train_dir}/spk2utt
utils/utt2spk_to_spk2utt.pl ${valid_dir}/utt2spk > ${valid_dir}/spk2utt
utils/utt2spk_to_spk2utt.pl ${test_dir}/utt2spk > ${test_dir}/spk2utt
echo "finished making text, wav.scp, utt2spk and spk2utt."