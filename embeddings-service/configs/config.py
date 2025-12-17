import yaml

with open("configs/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

print(CONFIG)