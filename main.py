from qcit.config import set_torch_config
set_torch_config()
from qcit.training import start_training
import yaml

with open("/notebooks/params.yaml", "r") as f:
    args = yaml.safe_load(f)

start_training(args)
