def get_constants():
    return {
        'len_seed': 100,
        'random_seed': 42,
        'mnist': {
            'width': 28,
            'height': 28,
            'color_mode': 'grayscale',
            'num_color_channels': 1,
            'batch_size': 100,
            'pixel_min': 0,
            'pixel_max': 255,
            'local': 'data/mnist',
            'colab': '/content/mnist'
        },
        'scapes': {
            'width': 1280,
            'height': 720,
            'color_mode': 'rgb',
            'num_color_channels': 3,
            'batch_size': 16,
            'pixel_min': 0,
            'pixel_max': 255,
            'local': 'data/scapes',
            'colab': '/content/scapes'
        }
    }