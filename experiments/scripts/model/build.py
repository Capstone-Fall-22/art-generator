from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from config import get_config
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose, BatchNormalization, Conv2D, LeakyReLU, Flatten, UpSampling2D, Resizing

def build_dcgan(dataset_name):
    weight_initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)

    config = get_config(dataset_name)

    generator_specs = config['dataset']['dcgan']['generator']

    input_layers = [
        Input(shape=(config['len_seed'],)),
        Dense(
            generator_specs['initial_width'] * generator_specs['initial_height'] * generator_specs['initial_num_filters'], 
            activation='relu', 
            kernel_initializer=weight_initializer
        ),
        Reshape((generator_specs['initial_height'], generator_specs['initial_width'], generator_specs['initial_num_filters']))
    ]

    hidden_convolutional_layers = []
    for i in range(generator_specs['num_hidden_conv_layers']):

        num_filters = generator_specs['initial_num_filters'] // (2**(i + 1))
        if num_filters < 32:
            num_filters = 32

        layers = [
            Conv2DTranspose(
                # Input layer has initial_num_filters, each hidden layer has half the
                # number of filters of the previous hidden layer
                num_filters, 
                kernel_size=3, 
                strides=2,
                padding='same', 
                activation='relu', 
                kernel_initializer=weight_initializer
            ),
            BatchNormalization(momentum=0.8)
        ]
        hidden_convolutional_layers.extend(layers)

    output_layers = [
        Resizing(
            config['dataset']['height'],
            config['dataset']['width']
        ),
        Conv2D(
            config['dataset']['num_color_channels'], 
            kernel_size=3, 
            padding='same', 
            activation='tanh', 
            kernel_initializer=weight_initializer
        )
    ]

    generator = Sequential(input_layers + hidden_convolutional_layers + output_layers)

    discriminator_specs = config['dataset']['dcgan']['discriminator']

    input_layers = [
        Input(
            shape=(
                config['dataset']['height'], 
                config['dataset']['width'], 
                config['dataset']['num_color_channels']
            )
        ),
        Conv2D(
            discriminator_specs['initial_num_filters'], 
            kernel_size=3, 
            strides=2, 
            padding='same'
        ),
        LeakyReLU(0.2),
    ]

    hidden_layers = []
    for i in range(discriminator_specs['num_hidden_conv_layers']):
        layers = [
            Conv2D(
                discriminator_specs['initial_num_filters'] * (2**(i + 1)),
                kernel_size=3, 
                strides=2, 
                padding='same'
            ),
            LeakyReLU(0.2)
        ]
        hidden_layers.extend(layers)

    output_layers = [
        Flatten(),
        Dense(1, activation='sigmoid')
    ]

    discriminator = Sequential(input_layers + hidden_layers + output_layers)

    return generator, discriminator

def build_toy_model(dataset_name):
    constants = get_constants()

    initial_height = constants[dataset_name]['height'] / 4
    initial_width = constants[dataset_name]['width'] / 4

    if not initial_height.is_integer() or not initial_width.is_integer():
        print('Width and height must be divisible by 4')
        return

    initial_height = int(initial_height)
    initial_width = int(initial_width)

    generator = Sequential([
        Dense(initial_height * initial_width * 64, input_shape=(constants['len_seed'],)),
        Reshape((initial_height, initial_width, 64)),

        UpSampling2D(),
        Conv2D(32, kernel_size=3, padding='same', activation='relu'),

        UpSampling2D(),
        Conv2D(16, kernel_size=3, padding='same', activation='relu'),
        
        Conv2D(constants[dataset_name]['num_color_channels'], kernel_size=3, padding='same', activation='tanh')
    ])

    discriminator = Sequential([
        Conv2D(
            16, 
            kernel_size=3, 
            input_shape=(
                constants[dataset_name]['height'],
                constants[dataset_name]['width'],
                constants[dataset_name]['num_color_channels']
            ), 
            padding='same', 
            activation='relu'
        ),
        Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'),
        Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    return generator, discriminator

def build_model(model_name, dataset_name):
    if model_name == 'dcgan':
        return build_dcgan(dataset_name)
    elif model_name == 'toy_model':
        return build_toy_model(dataset_name)
    else:
        print('Invalid model name')
        return