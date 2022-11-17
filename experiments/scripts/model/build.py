from silence_tensorflow import silence_tensorflow

silence_tensorflow()
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Dropout,
    BatchNormalization,
    Activation,
    ZeroPadding2D,
    LeakyReLU,
    UpSampling2D,
    Conv2D,
    Conv2DTranspose,
    ReLU,
    Resizing,
)


def build_generator(config):
    # ===========================================================================
    # Input Layer
    # ===========================================================================
    input_config = config["input_layer"]
    input_layers = [
        Input(shape=(input_config["len_seed"],)),
        Dense(
            input_config["height"]
            * input_config["width"]
            * input_config["num_filters"],
        ),
        Reshape(
            (input_config["height"], input_config["width"], input_config["num_filters"])
        ),
        ReLU(negative_slope=input_config["relu_negative_slope"]),
    ]

    if input_config["batch_norm"]:
        input_layers.append(BatchNormalization())

    current_image_shape = [input_config["height"], input_config["width"]]
    # ===========================================================================
    # Hidden Layers
    # ===========================================================================
    hidden_config = config["hidden_layers"]

    # Check that config has matching number of filter_counts and kernel_sizes
    assert len(hidden_config["filter_counts"]) == len(hidden_config["kernel_sizes"])
    # Check that only one of upsample or deconvolution is True
    assert not (hidden_config["upsample"] == hidden_config["deconvolution"])

    # Prepare weight initializer
    if hidden_config["weight_init"]:
        weight_initializer = RandomNormal(
            mean=hidden_config["weight_init_mean"],
            stddev=hidden_config["weight_init_stddev"],
        )
    else:
        weight_initializer = "glorot_uniform"  # Default initializer

    hidden_layers = []
    if hidden_config["upsample"]:
        for i in range(len(hidden_config["filter_counts"])):
            hidden_layers.extend(
                [
                    UpSampling2D(),
                    Conv2D(
                        hidden_config["filter_counts"][i],
                        kernel_size=hidden_config["kernel_sizes"][i],
                        padding="same",
                        kernel_initializer=weight_initializer,
                    ),
                ]
            )
            if hidden_config["batch_norm"]:
                hidden_layers.append(BatchNormalization())
            current_image_shape = [dimension * 2 for dimension in current_image_shape]
    if hidden_config["deconvolution"]:
        for i in range(len(hidden_config["filter_counts"])):
            hidden_layers.append(
                Conv2DTranspose(
                    hidden_config["filter_counts"][i],
                    kernel_size=hidden_config["kernel_sizes"][i],
                    strides=hidden_config["stride"],
                    padding="same",
                    kernel_initializer=weight_initializer,
                )
            )
            if hidden_config["batch_norm"]:
                hidden_layers.append(BatchNormalization())
            current_image_shape = [dimension * 2 for dimension in current_image_shape]

        hidden_layers.append(ReLU(negative_slope=hidden_config["relu_negative_slope"]))

        if hidden_config["batch_norm"]:
            hidden_layers.append(BatchNormalization())

    # ===========================================================================
    # Output Layer
    # ===========================================================================
    output_config = config["output_layer"]
    output_layers = []
    if (
        current_image_shape[0] > output_config["height"]
        or current_image_shape[1] > output_config["width"]
    ):
        output_layers.append(Resizing(output_config["height"], output_config["width"]))

    output_layers.append(
        Conv2D(
            output_config["num_color_channels"],
            kernel_size=output_config["kernel_size"],
            padding="same",
            activation=output_config["activation"],
            kernel_initializer=weight_initializer,
        )
    )

    generator = Sequential(input_layers + hidden_layers + output_layers)

    return generator


def build_discriminator(config):
    # ===========================================================================
    # Input Layer
    # ===========================================================================
    input_config = config["input_layer"]
    input_layers = [
        Input(
            shape=(
                input_config["height"],
                input_config["width"],
                input_config["num_color_channels"],
            )
        )
    ]

    # ===========================================================================
    # Hidden Layers
    # ===========================================================================
    hidden_config = config["hidden_layers"]
    hidden_layers = []

    # Check that config has matching number of filter_counts and kernel_sizes
    assert len(hidden_config["filter_counts"]) == len(hidden_config["kernel_sizes"])

    # Check that config has machine number of filter_counts and strides
    assert len(hidden_config["filter_counts"]) == len(hidden_config["strides"])

    for i in range(len(hidden_config["filter_counts"])):
        hidden_layers.extend(
            [
                Conv2D(
                    hidden_config["filter_counts"][i],
                    kernel_size=hidden_config["kernel_sizes"][i],
                    strides=hidden_config["strides"][i],
                    padding="same"
                ),
                ReLU(negative_slope=hidden_config["relu_negative_slope"]),
                Dropout(hidden_config["dropout"])
            ]
        )
    
    # ===========================================================================
    # Output Layer
    # ===========================================================================
    output_config = config["output_layer"]
    output_layers = [
        Flatten(),
        Dense(output_config["num_units"], activation=output_config["activation"])
    ]

    discriminator = Sequential(input_layers + hidden_layers + output_layers)

    return discriminator



def load_model(config):
    generator = build_generator(config["generator"])
    discriminator = build_discriminator(config["discriminator"])
    # print(generator.summary(), discriminator.summary())
    return generator, discriminator

def get_architecture(config):
    generator = build_generator(config["generator"])
    discriminator = build_discriminator(config["discriminator"])
    return generator.summary(), discriminator.summary()
