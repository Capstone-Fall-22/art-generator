import json, os, shutil
import tensorflow as tf
from scripts.data.load import load_dataset
from scripts.model.build import load_model
from scripts.model.hyperparameters import load_optimizers
from scripts.model.train import train

'''
Reads a JSON file with the experiment config
'''
def load_config(config_name, config_type):
    config_path = f"config/{config_type}/{config_name}.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    return config

'''
Prepares a dictionary with everything needed to run experiment, then trains the models

Dataset name - Name of dataset config file (without .json) to load
Model name - Nmae of model config file (without .json) to load
Experiment name - Name of folder to include output in
Number of examples to generate - How many images we generate after each epoch to see the results
Colab - whether the experiment is running on a colab notebook (affects where the data gets loaded from)
'''
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