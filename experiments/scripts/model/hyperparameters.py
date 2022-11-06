from tensorflow.keras.optimizers import Adam

def get_dcgan_hyperparameters():
    return {
        'optimizer': Adam(0.0002, 0.5)
    }

def get_toy_model_hyperparameters():
    return {
        'optimizer': Adam(0.001)
    }