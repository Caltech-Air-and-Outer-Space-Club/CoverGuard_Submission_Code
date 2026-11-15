#!/bin/bash
# usage: run_matrix.sh "<variant:seed> ..." [parallel]   -- runs queue with N parallel workers, 3 threads each
cd "$(dirname "$0")"
printf '%s\n' $1 | xargs -P ${2:-3} -I{} bash -c 'v=${1%%:*}; s=${1##*:}; ../.venv/bin/python -W ignore train_v2.py --variant $v --seed $s --threads 3 $EXTRA > runs/log_${v}${TAG}_s${s}.txt 2>&1' _ {}
echo MATRIX_DONE >> runs/matrix_status.txt
