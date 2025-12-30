import time

def get_first_key_param(model: dict):
    for key, value in model.items():
        key_name = key
        param = value
        break

    return key_name, param

def generate_unique_id():
    return int(f"{int(time.time() * 1000) % 1000000:06d}")