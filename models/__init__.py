from .lmmcaps import LMMCAPS

def load_model(config):
    return LMMCAPS.from_config(config)