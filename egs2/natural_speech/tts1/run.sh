#!/bin/bash

if [ -z "$1" ]; then
    stage=-1
else
    stage=$1
fi

if [ -z "$2" ]; then
    stop_stage=100
else
    stop_stage=$2
fi

set -e
set -u
set -o pipefail

tts_task=gan_tts
expdir=
tag=
tts_stats_dir="${expdir}/tts_stats"
data_dir=
train_set="train"
valid_set="valid"
test_sets="test"
train_config=

ngpu=
multiprocessing_distributed=

