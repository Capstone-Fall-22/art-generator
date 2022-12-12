import  time, gc, os, shutil
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import matplotlib.pyplot as plt
from IPython import display
from scripts.model.loss import generator_loss, discriminator_loss
import numpy as np
import logging

'''
Samples the model's output

Generators len(test_seeds) images and displays them using matplotlib, then saves the figure to a file
'''
def generate_and_save_images(model, epoch, test_seeds, output_dir):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm doesn't get used in inference mode)
    predictions = model(test_seeds, training=False)
    predictions = predictions.numpy().astype("float32")
    print(np.average(predictions[0]), np.max(predictions[0]), np.min(predictions[0]))
    fig = plt.figure(figsize=(4, 4))
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)
    for i, image in enumerate(predictions):
        image = ((image - np.min(image)) * 255) / (np.max(image) - np.min(image))
        image = image.astype("uint8")
        plt.subplot(4, 4, i+1)
        plt.imshow(image)
        plt.axis('off')  

    plt.savefig(os.path.join(output_dir, 'images', 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.show()
    logger.setLevel(old_level)


'''
A tensorflow compiled function that defines the flow of a single training batch

@tf.function() means its a tensorflow compiled function which is precompiled before training and adds important optimizations for ML

    - Get input seeds for generating images
    - Use the generator to generate images
    - Have the discriminator predict whether the images are real or fake on a batch of real images and a batch of fake images
    - Compute the loss for the generator and discriminator
    - Applies backpropagation to the generator and discriminator based on loss
        - Computes gradients using gradient tape
        - Applies the gradients (updates model weights towards hopefully better performance)
'''
@tf.function()
def train_step(config, real_image_batch):
    input_noise_seeds = tf.random.normal([config['batch_size'], config['len_seed']])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image_batch = config["generator"](input_noise_seeds, training=True)

        y_pred_real = config["discriminator"](real_image_batch, training=True)
        y_pred_fake = config["discriminator"](generated_image_batch, training=True)

        gen_loss = generator_loss(y_pred_fake)
        disc_loss = discriminator_loss(y_pred_real, y_pred_fake)

    generator_gradient = gen_tape.gradient(gen_loss, config["generator"].trainable_variables)
    discriminator_gradient = disc_tape.gradient(disc_loss, config["discriminator"].trainable_variables)

    config["gen_optimizer"].apply_gradients(zip(generator_gradient, config["generator"].trainable_variables))
    config["disc_optimizer"].apply_gradients(zip(discriminator_gradient, config["discriminator"].trainable_variables))

    return gen_loss, disc_loss

'''
Main training loop which goes through all of the epochs of training

    - Starts by creating the folders for the experiment results
    - Validates certain config values
    - Creates a constant set of seeds so we can see how well the model does over time
    - For each epoch, split the dataset into batches and for each batch run a single training step
        - After it batch, it displays the loss of the generator and discriminator on the batch
    - At the end of each epoch
        - Sample the results of the generator and save the generator and discriminator for future use
'''
def train(config):
    gc.collect()
    output_dir = f"/content/{config['output_dir']}" if config["colab"] else config['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # if experiment folder exists, delete it and create a new one
    if os.path.exists(f"{output_dir}/{config['experiment_name']}"):
        shutil.rmtree(f"{output_dir}/{config['experiment_name']}")
    os.mkdir(f"{output_dir}/{config['experiment_name']}")
    os.mkdir(os.path.join(f"{output_dir}/{config['experiment_name']}", "images"))

    if config["num_examples_to_generate"] < 1:
        print('Number of examples to generate invalid (must be > 0)')
        return
  
    test_seeds = tf.random.normal([config["num_examples_to_generate"], config['len_seed']])

    for epoch in range(config["epochs"]):
        start = time.time()
        gen_loss, disc_loss = None, None

        for i, batch in enumerate(config["dataset"]):
            display.clear_output(wait=True)
            print(f'Loss for previous batch #{i}: Generator loss = {gen_loss}, Discriminator loss = {disc_loss}')
            print(f'Epoch # {epoch + 1}/{config["epochs"]}')
            print(f'Batch # {i + 1}')
            gc.collect()
            gen_loss, disc_loss = train_step(config, batch)
        
        generate_and_save_images(config["generator"], epoch + 1, test_seeds, os.path.join(output_dir, config["experiment_name"]))
        config["generator"].save(os.path.join(output_dir, config["experiment_name"], f"generator_epoch_{epoch + 1}"))
        config["discriminator"].save(os.path.join(output_dir, config["experiment_name"], f"discriminator_epoch_{epoch + 1}"))


        print(f'Time for epoch {epoch + 1} is {time.time()-start} sec')
  
    display.clear_output(wait=True)
