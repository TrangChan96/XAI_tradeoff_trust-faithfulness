import os
# import multiprocessing
import argparse
import json
from slurm.slurm_template import slurm_template
from datetime import datetime
import subprocess
import sys
# from utils.utils import *

# file for submitting multiple training tasks on the cluster
def main():
    if len(sys.argv) < 2:
        print("Please pass a config file as first argument and a job id as second argument to run the experiment.")
        exit(0)

    # get parameters
    params = json.load(open(sys.argv[1]))
    root_data = params["root_dataset"]
    path_model = params["path_model"]
    result_path = params["result_path"]
    path_bias_model = params["save_path_bias_model"]

    batch_size = params["batch_size"]
    num_epochs = params["epochs"]
    lambda_values = params["lambda"]
    fooling_methods = params["fooling_method"]
    basemethods = params["basemethod"]
    modifiers = params["modifiers"]
    bias = params["bias"]
    percentage = params["percentage"]
    sizes = params["window_size"]
    seeds = params["seed"]

    params_dict = {}
    c = 0
    for fooling_method in fooling_methods:
        for basemethod in basemethods:
            for modifier in modifiers:
                # for i in range(0, len(lambda_values), 2):
                for lambda_value in lambda_values:
                    for size in sizes:
                        for seed in seeds:
                            params_dict["fooling_method"] = fooling_method
                            # params_dict["lambda"] = lambda_values[i:i+2]
                            # exit()
                            params_dict["lambda"] = [lambda_value]
                            params_dict["basemethod"] = basemethod
                            params_dict["modifiers"] = [modifier]
                            now = datetime.now()
                            current_time = now.strftime("%H_%M_%S")
                            # params_dict["save_path_bias_model"] = os.path.join(fooling_method,
                            #                                                    "{}-{}".format(basemethod, modifier),
                            #                                                    '%s' % (lambda_value,),
                            #                                                    "biased_model_{}.pth".format(current_time))
                            params_dict["save_path_bias_model"] = path_bias_model
                            params_dict["result_path"] = result_path
                            params_dict["root_dataset"] = root_data
                            params_dict["path_model"] = path_model
                            params_dict["epochs"] = num_epochs
                            params_dict["batch_size"] = batch_size
                            params_dict["bias"] = bias
                            params_dict["percentage"] = percentage
                            params_dict["window_size"] = size
                            params_dict["seed"] = seed

                            folder_name = "folder_params"
                            os.makedirs(folder_name, exist_ok=True)
                            filename = f"params_{fooling_method}_{basemethod}_{modifier}_{lambda_value}_{size}_{seed}.json"
                            file_path = os.path.join(folder_name, filename)
                            with open(file_path, 'w') as f:
                                json.dump(params_dict, f)

                            # precmd = "cd master-thesis"
                            cmd = f"singularity exec --nv /common/singularityImages/TCML-Cuda11_2_TF2_6_0_PT_1_10_0.simg " \
                                      f"python3 ./master-thesis/ExplanationManipulation.py $SLURM_JOB_ID {file_path}"
                            # os.system(cmd)
                            slurm_string = slurm_template(partition="day", time="1-0", mail_type="END", command=cmd)
                            filestring = slurm_string.generate_filestring()

                            sbatch_file = os.path.join("manipulation.sbatch")
                            with open(sbatch_file, "w+", encoding='utf-8', newline='\n') as f:
                                f.write(filestring)
                            try:
                                command = "sbatch {}".format(sbatch_file)
                                # subprocess.call(command)#, shell=True)
                                os.system(command)
                            except:
                                print("Fail to batch")
                            # os.remove(filename)
                            os.remove(sbatch_file)

if __name__ == '__main__':
    main()
