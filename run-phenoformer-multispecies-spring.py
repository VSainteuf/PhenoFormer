from pathlib import Path
from subprocess import call

import configs.PROBLEM_CONFIG as cfg
from configs.RUN_CONFIGS import (datasplit_configs, model_configs,
                                 training_configs)

"""
This script can run the 40 folds of each multi-task configuration of PhenoFormer 
for the spring phenophases (Table 6). Simply uncomment the configuration that you would
like to run, and specify the path of the dataset folder, and the path of where you 
would like the output to be written. 
"""

data_folder = "path/to/data"
save_dir = "path/to/outputdir"


## CONFIG 1 : Multi-species model (b)
model_config = "phenoformer_default"
training_config = "default"
target = (
    "LU+NE"  # predict all leaf unfolding / needle emergence phases at the same time
)

## CONFIG 2 : Multi-species model + flowering (c)
# model_config = "phenoformer_default"
# training_config = "default"
# target = 'LU+NE+FL' # predict all leaf unfolding / needle emergence AND flowering phases at the same time

## CONFIG 3 : Multi-species model + static site data (d)
# model_config = "phenoformer_staticdata"
# training_config = "default"
# target = 'LU+NE' # predict all leaf unfolding / needle emergence phases at the same time

## CONFIG 4 : Multi-species model + static site data + flowering (e)
# model_config = "phenoformer_staticdata"
# training_config = "default"
# target = 'LU+NE+FL' # predict all leaf unfolding / needle emergence AND flowering phases at the same time


to_do_list = ["uniformly_rdm", "rdm_spatial", "rdm_temporal", "structured_temporal"]


for split_config in to_do_list:
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
    cmd.extend(["--task_tag", "multispecies"])

    unique_id = f"MultiTask-{model_config}-{target.replace(':','')}-{split_config}-{training_config}"
    cmd.extend(["--cross_val_id", unique_id])
    call(cmd)
