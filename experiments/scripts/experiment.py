import json, os, shutil
import tensorflow as tf
from scripts.data.load import load_dataset
from scripts.model.build import load_model
from scripts.model.hyperparameters import load_optimizers
from scripts.model.train import train

def load_config(config_name, config_type):
    config_path = f"config/{config_type}/{config_name}.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def run_experiment(dataset_name, model_name, experiment_name, num_examples_to_generate=1, epochs=10, colab=False):
    # Builds the model, traings for some number of epochs, and saves the model and sample output at each epoch
    dataset_config = load_config(config_name=dataset_name, config_type="dataset")
    model_config = load_config(config_name=model_name, config_type="model")

    dataset = load_dataset(dataset_config, colab=colab)
    generator, discriminator = load_model(model_config)
    gen_optimizer, disc_optimizer = load_optimizers(model_config)

    train_config = {
        "epochs": epochs,
        "output_dir": "results",
        "experiment_name": experiment_name,
        "dataset": dataset,
        "generator": generator,
        "discriminator": discriminator,
        "gen_optimizer": gen_optimizer,
        "disc_optimizer": disc_optimizer,
        "num_examples_to_generate": num_examples_to_generate,
        "batch_size": dataset_config["batch_size"],
        "len_seed": model_config["generator"]["input_layer"]["len_seed"],
        "colab": colab
    }

    train(train_config)

    return generator, discriminator



# TODO Implement this
def continue_experiment(dataset_name, model_name, generator_path, discriminator_path, experiment_name, num_examples_to_generate=1, epochs=10, colab=False):
    # Builds the model, traings for some number of epochs, and saves the model and sample output at each epoch
    dataset_config = load_config(config_name=dataset_name, config_type="dataset")
    model_config = load_config(config_name=model_name, config_type="model")

    dataset = load_dataset(dataset_config, colab=colab)

    generator = tf.keras.models.load_model(generator_path)
    discriminator = tf.keras.models.load_model(discriminator_path)

    gen_optimizer, disc_optimizer = load_optimizers(model_config)

    train_config = {
        "epochs": epochs,
        "output_dir": "results",
        "experiment_name": experiment_name,
        "dataset": dataset,
        "generator": generator,
        "discriminator": discriminator,
        "gen_optimizer": gen_optimizer,
        "disc_optimizer": disc_optimizer,
        "num_examples_to_generate": num_examples_to_generate,
        "batch_size": dataset_config["batch_size"],
        "len_seed": model_config["generator"]["input_layer"]["len_seed"],
        "colab": colab
    }

    train(train_config)

    return generator, discriminator
