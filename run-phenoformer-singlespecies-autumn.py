from pathlib import Path
from subprocess import call

import configs.PROBLEM_CONFIG as cfg
from configs.RUN_CONFIGS import (datasplit_configs, model_configs,
                                 training_configs)

"""
This script can run the 40 folds of the single-task variants (single-species (a,f)) of PhenoFormer 
for the autumn phenophases (Table 9). By default it will run the experiment for all of the four 
dataset splits, but you can change that by changing the to_do_list. 
Simply uncomment the configuration that you would like to run, and specify the path 
of the dataset folder, and the path of where you would like the output to be written. 
"""

data_folder = "path/to/data"
save_dir = "path/to/outputdir"


# CONFIG 1 : Single-species model (a)
model_config = "phenoformer_default"
training_config = "default"
spring_as_input = False

# CONFIG 2 : Single-species model with spring phenophase date as additonal input (f)
# model_config = "phenoformer_default"
# training_config = "default"
# spring_as_input = True


to_do_list = ["uniformly_rdm", "rdm_spatial", "rdm_temporal", "structured_temporal"]


for split_config in to_do_list:
    for target_long in cfg.autumn_phenophases:
        print(target_long)
        species, phase = target_long.split(":")
        target = f"{cfg.species_short[species]}:{cfg.phases_short[phase]}"

        cmd = ["python", str(Path(__file__).resolve().parent / "cross_val_train.py")]

        run_config = {
            **model_configs[model_config],
            **training_configs[training_config],
            **datasplit_configs[split_config],
        }

        for k, v in run_config.items():
            if isinstance(v, bool):
                if v:
                    cmd.extend([f"--{k}"])
            elif v is not None:
                cmd.extend([f"--{k}", str(v)])

        cmd.extend(["--data_folder", data_folder])
        cmd.extend(["--save_dir", save_dir])
        cmd.extend(["--target", target])
        cmd.extend(["--model_tag", model_config])
        cmd.extend(["--config_tag", training_config])
        cmd.extend(["--task_tag", "singlespecies"])

        ## Add spring as input
        if spring_as_input:
            ip = "LU" if cfg.phases_short[phase] == "LC" else "NE"
            input_phenophase = f"{cfg.species_short[species]}:{ip}"
            cmd.extend(["--input_phases", input_phenophase])

        unique_id = f"SingleTask-{model_config}-{target.replace(':','#')}-{split_config}-{training_config}"
        cmd.extend(["--cross_val_id", unique_id])
        call(cmd)
