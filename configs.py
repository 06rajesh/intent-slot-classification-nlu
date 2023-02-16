import json
import os.path
import ast

from typing import List
from pathlib import Path

class Config:
    intents_list: List
    tags_list: List
    max_sentence_length: int
    device:str
    batch_size:int
    num_workers:int
    dropout_rate: float
    learning_rate: float
    weight_decay: float
    start_epoch: int
    epochs: int
    ignore_index:int
    slot_loss_coef:float
    output_dir:str
    version:str

    def __init__(self,
                 device:str ='cpu',
                 batch_size:int = 2,
                 num_workers:int = 1,
                 max_sentence_length=50,
                 dropout: int = 0.20,
                 learning_rate: float = 0.0001,
                 weight_decay: float = 0.0005,
                 start_epoch: int = 0,
                 epochs: int = 10,
                 ignore_index:int = -100,
                 slot_loss_coef:float = 1.0,
                 output_dir:str = './pretrained/',
                 version:str = 'v.0.1'
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_sentence_length = max_sentence_length
        self.dropout_rate = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.ignore_index = ignore_index
        self.slot_loss_coef = slot_loss_coef
        self.output_dir = output_dir
        self.version = version

    @classmethod
    def from_pretrained(self, config_path: str):

        config_path = Path(config_path)

        if os.path.isdir(config_path):
            config_path = config_path / 'config.json'

        default_conf = self()
        default_conf.tags_list = []
        default_conf.intents_list = []

        # Opening JSON file
        f = open(config_path)

        configs = json.load(f)

        f.close()

        for item in configs:
            val = getattr(default_conf, item)
            config_val = configs[item]
            if isinstance(val, bool):
                config_val = True if configs[item].capitalize() == 'True' else False
            elif isinstance(val, int):
                config_val = int(configs[item])
            elif isinstance(val, float):
                config_val=  float(configs[item])
            elif isinstance(val, List):
                config_val = ast.literal_eval(configs[item])

            setattr(default_conf, item, config_val)

        return default_conf

    def save_config(self, path:str=None):

        config = {}
        exclude = []

        for attr, value in self.__dict__.items():
            if attr not in exclude:
                config[attr] = str(value)

        if not path:
            path = self.output_dir
            if self.version != "":
                path = path + "/" + self.version

        path = Path(path)
        target_file = path / 'config.json'

        with open(target_file, 'w') as f:
            json.dump(config, f, indent=2)