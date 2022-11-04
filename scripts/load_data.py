import tensorflow as tf

def load_dataset(name, batch_size, image_dimensions):
    return tf.keras.utils.image_dataset_from_directory(
        directory=f'/content/{name}', 
        labels=None, 
        color_mode='rgb', 
        batch_size=batch_size, 
        image_size=(image_dimensions[1], image_dimensions[0]), 
        shuffle=True, 
        seed=42, 
        validation_split=None, 
        subset=None, 
        interpolation='bilinear', 
        follow_links=False
    )
