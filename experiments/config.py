from tensorflow.keras.optimizers import Adam

def get_config(dataset_name=None, model_name=None):
    if dataset_name:
        return 
    dataset_configs = {
        'mnist': {
            'toy_model': {
                'num_hidden_layers': 0,
                'initial_num_filters': 0
            },
            'dcgan': {
                'generator': {
                    'initial_num_filters': 256,
                    'num_hidden_conv_layers': 2,
                    'initial_width': 7,
                    'initial_height': 7,
                    'len_seed': 7 * 7,
                    'optimizer': Adam(0.0002, 0.5),
                    'double_conv': False
                },
                'discriminator': {
                    'initial_num_filters': 128,
                    'num_hidden_conv_layers':  2,
                    'optimizer': Adam(0.0002, 0.5)
                }
            },

        'scapes': {
            'toy_model': {
                'num_hidden_layers': 0,
                'initial_num_filters': 0
            },
            'dcgan': {
                'generator': {
                    'initial_num_filters': 1024,
                    'num_hidden_conv_layers': 7,
                    'initial_width': 10,
                    'initial_height': 10,
                    'len_seed': 100,
                    'optimizer': Adam(0.0002, 0.5),
                    'double_conv': False, # Double conv appears to hurt performance
                    'filter_halving_occurance': 2, # How many conv layers to have before halving the number of filters
                    'minimum_num_filters': 64
                },
                'discriminator': {
                    'initial_num_filters': 128,
                    'num_hidden_conv_layers':  3,
                    'optimizer': Adam(0.0002, 0.5)
                }            
            },
            

        }
    }

    config = {
        'random_seed': 42,
        'dataset': dataset_configs[dataset_name]
    }

    config['dataset']['name'] = dataset_name

    return config