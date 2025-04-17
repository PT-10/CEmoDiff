import yaml

def get_config(yaml_path):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return config