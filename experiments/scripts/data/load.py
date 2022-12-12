from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import tensorflow as tf

'''
Loaded the dataset using tensorflow's image_dataset_from_directory based on the experiment configuration.
Once the data is loaded, it gets preprocessed based on the config.
    - Normalization: Rescales pixels values to [-1, 1]
    - Batch: Trains the model in batches of examples, instead of one at a time or all of them at once
    - Prefetch: Prepares the next batch while the current batch is being processed by the model

The colab argument determines where we look for the dataset, defined in the config.
'''
def load_dataset(config, colab=False):
    path_location = "colab" if colab else "local"

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=config["path"][path_location],
        labels=None,
        color_mode=config["color_mode"],
        image_size=(config["height"], config["width"]),
        batch_size=None,
        shuffle=config["shuffle"],
        seed=config["random_seed"],
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )

    if config["normalize"]:
        normalization_factor = (config["pixel_max"] + config["pixel_min"]) / 2
        # Rescale to [-1, 1]
        dataset = dataset.map(
            lambda x: (x - normalization_factor) / normalization_factor
        )

    if config["batch"]:
        dataset = dataset.batch(config["batch_size"], drop_remainder=True)

    if config["prefetch"]:
        dataset = dataset.prefetch(config["prefetch"])

    return dataset
