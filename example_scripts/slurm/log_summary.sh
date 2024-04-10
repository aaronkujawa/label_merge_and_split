#!/usr/bin/env bash
cd logs

n_last=${1}

if [ -z "$n_last" ]; then
    n_last=1
fi

echo -e "--------log.err-------------\n"; ls *.err | tail -n $n_last | head -n 1 | xargs cat | tail;  
echo -e "\n\n--------log.out-----------------\n"; ls *.out | tail -n $n_last | head -n 1 | xargs cat | tail; 
echo -e "\n-----------squeue------------\n"; squeue -u aku20
