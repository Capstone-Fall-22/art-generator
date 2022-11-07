from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.layers import Rescaling, Subtract, Normalization
from config import get_config

def load_dataset(
        dataset_name, 
        normalize=False,
        batch=False, 
        colab=False
    ):
    config = get_config(dataset_name)
    dataset_path = config['dataset']['path']['colab'] if colab else config['dataset']['path']['local']
    
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=dataset_path, 
        labels=None, 
        color_mode=config['dataset']['color_mode'], 
        image_size=(config['dataset']['height'], config['dataset']['width']), 
        batch_size=None,
        shuffle=True, 
        seed=config['random_seed'], 
        validation_split=None, 
        subset=None, 
        interpolation='bilinear', 
        follow_links=False
    )

    if normalize:
        normalization_factor = (config['dataset']['pixel_max'] + config['dataset']['pixel_min']) / 2
        # Rescale to [-1, 1]
        dataset = dataset.map(lambda x: (x - normalization_factor)/normalization_factor)

    if batch:
        dataset = dataset.batch(config['dataset']['batch_size'])

    return dataset