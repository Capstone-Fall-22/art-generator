import  time, gc, os, shutil
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import matplotlib.pyplot as plt
from IPython import display
from scripts.model.loss import generator_loss, discriminator_loss
from scripts.data.visualization import display_images
import numpy as np
import logging

def generate_and_save_images(model, epoch, test_seeds, output_dir):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_seeds, training=False)
    predictions = predictions.numpy()
    fig = plt.figure(figsize=(4, 4))
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)
    for i, image in enumerate(predictions):
        image = ((image - np.min(image)) * 255) / (np.max(image) - np.min(image))
        # print(np.average(image), np.max(image), np.min(image))
        plt.subplot(4, 4, i+1)
        plt.imshow(image)
        plt.axis('off')  

    plt.savefig(os.path.join(output_dir, 'images', 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.show()
    logger.setLevel(old_level)

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
