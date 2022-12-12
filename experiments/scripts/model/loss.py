import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

'''
Defines the loss function for the generator

loss = how many fake images were classfied as real by the discriminator (the higher the better for the generator)
'''
def generator_loss(y_pred_fake):
    binary_cross_entropy = BinaryCrossentropy()
    return binary_cross_entropy(tf.ones_like(y_pred_fake), y_pred_fake)


'''
Defines the loss function for the discriminator

loss = how many fake images were classified as fake + how many real images were classified as real (the lower the better for the generator)
'''
def discriminator_loss(y_pred_real, y_pred_fake):
    binary_cross_entropy = BinaryCrossentropy()
    real_loss = binary_cross_entropy(tf.ones_like(y_pred_real), y_pred_real)
    fake_loss = binary_cross_entropy(tf.zeros_like(y_pred_fake), y_pred_fake)
    return real_loss + fake_loss
