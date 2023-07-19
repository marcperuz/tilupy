_default_config = dict(language='english')

class Config(dict):
    
    def __init__(self, **kwargs):
        for key in kwargs:
            self[key] = kwargs[key]
            
    def update(self, **kwargs):
        for key in kwargs:
            self[key] = kwargs[key]
            
config = Config(**_default_config)

def set_config(**kwargs):
    config.update(**kwargs)
    
    if 'language' in kwargs:
        import tilupy.notations
        tilupy.notations.set_labels()
    
#import tilupy.notations
    