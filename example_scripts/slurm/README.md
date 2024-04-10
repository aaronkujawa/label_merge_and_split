These scripts can run a python script in parallel via slurm.
The python script only takes an integer as input.
The integer is used by the python script to select the correct arguments for the current run,
for example the input/output paths. 

Requirements can be installed on the slurm system by:
	a) module load miniconda
	b) conda activate local
	c) conda install <packagename>
	
	d) to install CPU-only pytorch use: conda install pytorch torchvision torchaudio cpuonly -c pytorch

To submit the jobs, run:
python tom2_submit_parallel_script.py calculate_distance_matrix_for_individual_label_file.py 3

The last integer is the number of jobs to run in total, so the last job will be run with index integer (3-1).


The latest error files can be monitored by running the log_summary.sh

