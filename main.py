import yaml
from qcit.training import start_training

with open("/notebooks/params.yaml", "r",  encoding="utf-8") as f:
    args = yaml.safe_load(f)

start_training(args)
