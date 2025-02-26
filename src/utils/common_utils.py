import yaml

def write_yaml(data, file_path):
    """Writes dictionary data to a YAML file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        raise RuntimeError(f"Error writing YAML file {file_path}: {e}")


def read_yaml(file_path):
    """Reads and returns data from a YAML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if data is not None else {} 
    except Exception as e:
        raise RuntimeError(f"Error reading YAML file {file_path}: {e}")
