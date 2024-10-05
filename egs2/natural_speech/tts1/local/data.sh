#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=2

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


train_set=train
valid_set=valid
test_set=test

org_wav_dir=$1


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    local/data_prep.sh "${org_wav_dir}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    # make validation and test sets
    # utils/copy_data_dir.sh "data/${spk}_parallel100" "data/${train_set}"
    utils/subset_data_dir.sh --per-spk "data" 500 "data/${valid_set}"
    utils/subset_data_dir.sh --per-spk "data" 500 "data/${test_set}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
