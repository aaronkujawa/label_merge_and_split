import os
import sys
import subprocess

sys.path.append(os.getcwd())

sbatch_script_path = "./tom2_sbatch_script_taskid.sh"
python_script_path = sys.argv[1]  # path to the python script to be executed (takes 1 argument: the task id --taskid )
nb_cases = int(sys.argv[2])  # total number of taskid to be submitted

nb_of_active_jobs = 8
nb_of_submitted_jobs_limit = 99999


nb_jobs_to_submit_at_once = min(nb_of_submitted_jobs_limit, nb_cases)
array_indeces_str = f"0-{nb_jobs_to_submit_at_once - 1}%{nb_of_active_jobs}"

print(f"Number of cases is {nb_cases}")
print(f"array_indeces_str is {array_indeces_str}")

# Replace the following line with the actual command for submitting the job.
sbatch_command = f"sbatch -a {array_indeces_str} {sbatch_script_path} {python_script_path} {str(nb_of_submitted_jobs_limit)}"
print("\n\nsbatch_command =", sbatch_command)
#os.system(sbatch_command)
subprocess.call(sbatch_command, shell=True)

