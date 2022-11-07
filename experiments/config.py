from tensorflow.keras.optimizers import Adam

def get_config(dataset_name):
    dataset_configs = {
        'mnist': {
            'toy_model': {
                'num_hidden_layers': 0,
                'initial_num_filters': 0
            },
            'dcgan': {
                'generator': {
                    'initial_num_filters': 128,
                    'num_hidden_conv_layers': 2,
                    'initial_width': 7,
                    'initial_height': 7,
                    'optimizer': Adam(0.0002, 0.5)
                },
                'discriminator': {
                    'initial_num_filters': 128,
                    'num_hidden_conv_layers':  1,
                    'optimizer': Adam(0.0002, 0.5)
                }
            },
            'path': {
                'local': 'data/mnist',
                'colab': '/content/mnist'
            },
            'batch_size': 100,
            'width': 28,
            'height': 28,
            'color_mode': 'grayscale',
            'num_color_channels': 1,
            'pixel_min': 0,
            'pixel_max': 255,
        },
        'scapes': {
            'toy_model': {
                'num_hidden_layers': 0,
                'initial_num_filters': 0
            },
            'dcgan': {
                'num_hidden_layers': 0,
                'initial_num_filters': 0
            },
            'path': {
                'local': 'data/scapes',
                'colab': '/content/scapes'
            },
            'batch_size': 16,
            'width': 1280,
            'height': 720,
            'color_mode': 'rgb',
            'num_color_channels': 3,
            'pixel_min': 0,
            'pixel_max': 255,
        }
    }

    config = {
        'len_seed': 100,
        'random_seed': 42,
        'dataset': dataset_configs[dataset_name]
    }

    config['dataset']['name'] = dataset_name

    return config