import tensorflow as tf
from scripts.constants import get_constants

def load_dataset(name, colab=False):
    constants = get_constants()
    seed = constants['random_seed']
    constants = constants[name]

    return tf.keras.utils.image_dataset_from_directory(
        directory=constants['colab'] if colab else constants['local'], 
        labels=None, 
        color_mode=constants[name]['color_mode'], 
        batch_size=constants['batch_size'], 
        image_size=(constants['width'], constants['height']), 
        shuffle=True, 
        seed=seed, 
        validation_split=None, 
        subset=None, 
        interpolation='bilinear', 
        follow_links=False
    )

print(load_dataset('mnist'))