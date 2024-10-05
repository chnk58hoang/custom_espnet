#!/usr/bin/env bash

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)



org_wav_dir=$1
data_dir=data

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <corpus_dir>"
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
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making text, wav.scp, utt2spk and spk2utt."