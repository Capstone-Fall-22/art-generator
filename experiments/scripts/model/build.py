from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from scripts.constants import get_constants
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose, BatchNormalization, Conv2D, LeakyReLU, Flatten, UpSampling2D

# Build model
def get_dcgan_specs(height, width):
    '''
    num_hidden_conv_layers:
        Equal to number of times we can evenly divide the width and height by 2.
        This is done this way since we use upsampling in each hidden layer which
        doubles the width and height.

    initial_num_filters 
        Derived from num_hidden_conv_layers. The goal is to have 8 filters in
        the last hidden layer. Since we halve the number of filters each time we
        upsample (as we do in each hidden conv layer) we can calculate the
        initial number of filters by taking 128 and doubling it for each hidden
        conv layer.
    '''

    num_hidden_conv_layers = 0
    while width > 4 and height > 4:
        width = width / 2
        height = height / 2
        if width.is_integer() and height.is_integer():
            num_hidden_conv_layers += 1
        else:
            break
    
    initial_width = width * 2
    initial_height = height * 2

    initial_num_filters = 8 * (2**num_hidden_conv_layers)
    
    specs = {
        'num_hidden_conv_layers': int(num_hidden_conv_layers),
        'initial_width': int(initial_width),
        'initial_height': int(initial_height),
        'initial_num_filters': int(initial_num_filters)
    }


    return specs

def build_dcgan(dataset_name):
    weight_initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)

    constants = get_constants()

    specs = get_dcgan_specs(constants[dataset_name]['height'], constants[dataset_name]['width'])
    print(specs)
    input_layers = [
        Input(shape=(constants['len_seed'],)),
        Dense(
            specs['initial_width'] * specs['initial_height'] * specs['initial_num_filters'], 
            activation='relu', 
            kernel_initializer=weight_initializer
        ),
        Reshape((specs['initial_height'], specs['initial_width'], specs['initial_num_filters']))
    ]

    hidden_convolutional_layers = []
    for i in range(specs['num_hidden_conv_layers']):
        layers = [
            Conv2DTranspose(
                # Input layer has initial_num_filters, each hidden layer has half the
                # number of filters of the previous hidden layer
                specs['initial_num_filters'] // (2**(i + 1)), 
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
        Conv2D(
            constants[dataset_name]['num_color_channels'], 
            kernel_size=3, 
            padding='same', 
            activation='tanh', 
            kernel_initializer=weight_initializer
        )
    ]

    generator = Sequential(input_layers + hidden_convolutional_layers + output_layers)

    initial_num_filters = 128

    input_layers = [
        Input(
            shape=(
                constants[dataset_name]['height'], 
                constants[dataset_name]['width'], 
                constants[dataset_name]['num_color_channels']
            )
        ),
        Conv2D(
            initial_num_filters, 
            kernel_size=3, 
            strides=2, 
            padding='same'
        ),
        LeakyReLU(0.2),
    ]

    hidden_layers = []
    for i in range(specs['num_hidden_conv_layers'] - 1):
        layers = [
            Conv2D(
            initial_num_filters * (2**(i + 1)),
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