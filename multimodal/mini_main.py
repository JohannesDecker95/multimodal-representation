from __future__ import print_function
import argparse
import yaml
import numpy as np
from logger import Logger
from trainers.selfsupervised import selfsupervised

# import warnings
# warnings.filterwarnings('error')
# torch.set_printoptions(profile="default")

import tensorboardX.x2num
from tensorboardX.x2num import check_nan as original_check_nan
# Monkey patching
def check_nan_patched(array):
    tmp = np.sum(array)
    if np.isnan(tmp) or np.isinf(tmp):
        raise ValueError('NaN or Inf found in input tensor.')
    return array

# Replace original function with patched version
tensorboardX.x2num.check_nan = check_nan_patched

# # Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    # Load the config file
    parser = argparse.ArgumentParser(description="Sensor fusion model")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--notes", default="", help="run notes")
    parser.add_argument("--dev", type=bool, default=False, help="run in dev mode")
    parser.add_argument(
        "--continuation",
        type=bool,
        default=False,
        help="continue a previous run. Will continue the log file",
    )
    args = parser.parse_args()

    # Add the yaml to the config args parse
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Merge configs and args
    for arg in vars(args):
        configs[arg] = getattr(args, arg)

    # Initialize the loggers
    logger = Logger(configs)

    # Initialize the trainer
    trainer = selfsupervised(configs, logger)

    trainer.train()
