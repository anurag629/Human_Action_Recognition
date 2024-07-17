import json

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.data_dir = config.get('data_dir')
        self.encoded_data_dir = config.get('encoded_data_dir')
        self.sample_per_vid = config.get('sample_per_vid')
        self.test_size = config.get('test_size')
        self.random_state = config.get('random_state')
        self.epochs = config.get('epochs')
        self.batch_size = config.get('batch_size')
        self.validation_split = config.get('validation_split')
        self.input_shape = config.get('input_shape')
        self.model_type = config.get('model_type')
        self.experiment_number = config.get('experiment_number')
