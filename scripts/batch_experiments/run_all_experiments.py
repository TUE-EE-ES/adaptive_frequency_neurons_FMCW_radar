import numpy as np

from src.batch_experiments.run_many_experiments import main as run_experiment

import os
import sys

configs_path = sys.argv[1]
save_path = sys.argv[2]

import os

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

for file in files(configs_path):
    filepath = os.path.join(configs_path, file)
    
    run_experiment(filepath, save_path)


