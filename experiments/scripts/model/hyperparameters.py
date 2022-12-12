from tensorflow.keras.optimizers import Adam, Adamax, Nadam


'''
Loads the optimizer specified in the config file for the generator and discriminator.
Affects the stability of the model's training

Options:
    (Most commonly used)
    - Adam
    
    (Derived from Adam)
    - Adamax
    - Nadam
'''
def load_optimizers(config):
    optimizer = None
    if config["generator"]["optimizer"]["name"] == "adam":
        optimizer = Adam
    elif config["generator"]["optimizer"]["name"] == "adamax":
        optimizer = Adamax
    elif config["generator"]["optimizer"]["name"] == "nadam":
        optimizer = Nadam

    gen_optimizer = optimizer(
        config["generator"]["optimizer"]["learning_rate"],
        config["generator"]["optimizer"]["beta_1"],
    )

    if config["discriminator"]["optimizer"]["name"] == "adam":
        optimizer = Adam
    elif config["discriminator"]["optimizer"]["name"] == "adamax":
        optimizer = Adamax
    elif config["discriminator"]["optimizer"]["name"] == "nadam":
        optimizer = Nadam
    
    disc_optimizer = optimizer(
        config["discriminator"]["optimizer"]["learning_rate"],
        config["discriminator"]["optimizer"]["beta_1"],
    )

    return gen_optimizer, disc_optimizer
