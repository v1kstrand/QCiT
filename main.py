import yaml

from qcit.config import set_torch_config
set_torch_config()
from qcit.training import start_training


with open("/notebooks/params.yaml", "r",  encoding="utf-8") as f:
    args = yaml.safe_load(f)

start_training(args)
