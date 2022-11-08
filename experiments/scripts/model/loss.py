import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

def generator_loss(y_pred_fake):
    binary_cross_entropy = BinaryCrossentropy()
    return binary_cross_entropy(tf.ones_like(y_pred_fake), y_pred_fake)

def discriminator_loss(y_pred_real, y_pred_fake):
    binary_cross_entropy = BinaryCrossentropy()
    real_loss = binary_cross_entropy(tf.ones_like(y_pred_real), y_pred_real)
    fake_loss = binary_cross_entropy(tf.zeros_like(y_pred_fake), y_pred_fake)
    return real_loss + fake_loss