def get_constants():
    return {
        'len_seed': 100,
        'random_seed': 42,
        'mnist': {
            'width': 28,
            'height': 28,
            'color_mode': 'grayscale',
            'batch_size': 100,
            'normalization_factor': 1/255,
            'local': 'data/mnist',
            'colab': '/content/mnist'
        },
        'scapes': {
            'width': 1280,
            'height': 720,
            'color_mode': 'rgb',
            'batch_size': 100,
            'normalization_factor': 1/255,
            'local': 'data/scapes',
            'colab': 'content/scapes'
        }
    }