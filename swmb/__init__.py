_default_config = dict(language='english')

class Config(dict):
    
    def __init__(self, **kwargs):
        for key in kwargs:
            self[key] = kwargs[key]
            
    def update(self, **kwargs):
        for key in kwargs:
            self[key] = kwargs[key]
            
config = Config(**_default_config)

def set_param(key, value):
    config.update(**{key: value})
    
import swmb.notations
    