"""
Script that runs the  N folds for a given model and
data configuration. It is meant to be called through the command line 
with the parameters of the configuration passed as arguments.
This script will then call the train.py script multiple times to 
run the folds.
"""

from copy import deepcopy
from pathlib import Path
from subprocess import call
from uuid import uuid4

from train import get_parser

n_fold = 40

if __name__ == "__main__":
    # Retrieve arguments
    parser = get_parser()
    args = parser.parse_args()

    # Get the full path to the script
    script_path = Path(__file__).resolve().parent

    # Prepare the command to launch train.py
    cmd = ["python", str(script_path / "train.py")]

    # Assign a common id to the 10 runs
    if args.cross_val_id is None:
        cmd.extend(["--cross_val_id", str(uuid4())])

    # Assign arguments
    for k, v in vars(args).items():
        if isinstance(v, bool):
            if v:
                cmd.extend([f"--{k}"])
        elif k == "save_dir":
            cmd.extend([f"--{k}", Path(v) / args.cross_val_id])
        elif v is not None:
            cmd.extend([f"--{k}", str(v)])

    # Run the folds with the same arguments
    for fold in range(1, n_fold + 1):
        fcmd = deepcopy(cmd)
        fcmd.extend(["--fold", str(fold)])
        call(fcmd)
