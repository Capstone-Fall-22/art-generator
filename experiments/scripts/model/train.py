import  time
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display

def generate_and_save_images(model, epoch, test_seeds):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_seeds, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

# @tf.function
def train_step(
        models,
        real_image_batch, 
        batch_size,
        len_seed
    ):
    input_noise_seeds = tf.random.normal([real_image_batch.numpy().shape[0], len_seed])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image_batch = models['generator']['model'](input_noise_seeds, training=True)

        y_pred_real = models['discriminator']['model'](real_image_batch, training=True)
        y_pred_fake = models['discriminator']['model'](generated_image_batch, training=True)

        gen_loss = models['generator']['loss'](y_pred_fake)
        disc_loss = models['discriminator']['loss'](y_pred_real, y_pred_fake)

        print(f'Generator loss: {gen_loss}, Discriminator loss: {disc_loss}', end='\r')

        generator_gradient = gen_tape.gradient(gen_loss, models['generator']['model'].trainable_variables)
        discriminator_gradient = disc_tape.gradient(disc_loss, models['discriminator']['model'].trainable_variables)

        models['generator']['optimizer'].apply_gradients(zip(generator_gradient, models['generator']['model'].trainable_variables))
        models['discriminator']['optimizer'].apply_gradients(zip(discriminator_gradient, models['discriminator']['model'].trainable_variables))

        return gen_loss, disc_loss

def train(models, dataset, epochs, len_seed, num_test_seeds):

    if num_test_seeds < 1:
        print('Number of test seeds invalid (must be > 0)')
        return
  
    test_seeds = tf.random.normal([num_test_seeds, len_seed])

    for epoch in range(epochs):
        start = time.time()
        gen_loss, disc_loss = None, None

        for i, batch in enumerate(dataset):
            display.clear_output(wait=True)
            print(f'Loss for previous batch #{i}: Generator loss = {gen_loss}, Discriminator loss = {disc_loss}')
            print(f'Epoch # {epoch + 1}/{epochs}')
            print(f'Batch # {i + 1}')
            gen_loss, disc_loss = train_step(models, batch, tf.shape(batch)[0], len_seed)
        
        generate_and_save_images(models['generator']['model'], epoch + 1, test_seeds)


        print(f'Time for epoch {epoch + 1} is {time.time()-start} sec')
  
    display.clear_output(wait=True)
    generate_and_save_images(models['generator']['model'], epochs, test_seeds)