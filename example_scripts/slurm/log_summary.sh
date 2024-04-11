#!/usr/bin/env bash
cd logs

n_last=${1}

if [ -z "$n_last" ]; then
    n_last=1
fi

err_file=`ls -rt *.err | tail -n $n_last | head -n 1`
echo -e "--------log.err-------------" $err_file "\n";
echo $err_file | xargs cat | tr '\r' '\n' | tail;  # tr removes carriage returns (for example from tqdm progress bar) because they cause problems with the watch command

out_file=`ls -rt *.out | tail -n $n_last | head -n 1`
echo -e "\n\n--------log.out-----------------" $out_file "\n";
echo $out_file | xargs cat | tr '\r' '\n' | tail;
echo -e "\n-----------squeue------------\n";
squeue -u aku20
