import yaml
import json

 
class Config(dict):
    def __init__(self, config_dict):
        super().__init__(**config_dict)
        self.__dict__ = self
        self.config_dict = config_dict

    @staticmethod
    def load_yaml(file_path):
        base_config = {}
        with open(file_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        base_config.update(config)
        return Config(base_config)

    @staticmethod
    def load_json(file_path):
        base_config = {}
        with open(file_path, encoding='utf-8') as f:
            config = json.load(f)
        base_config.update(config)
        return Config(base_config)
    
    def __repr__(self):
        return "{%s}"%', '.join("%r: %s"%p for p in self.items())