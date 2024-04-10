#!/usr/bin/env bash
cd logs

n_last=${1}

if [ -z "$n_last" ]; then
    n_last=1
fi

echo -e "--------log.err-------------\n";
ls *.err | tail -n $n_last | head -n 1 | xargs cat | tr '\r' '\n' | tail;  # tr removes carriage returns (for example from tqdm progress bar) because they cause problems with the watch command
echo -e "\n\n--------log.out-----------------\n";
ls *.out | tail -n $n_last | head -n 1 | xargs cat | tr '\r' '\n' | tail;
echo -e "\n-----------squeue------------\n";
squeue -u aku20
