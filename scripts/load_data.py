import tensorflow_datasets as tfds

def load_dataset(name_or_path, tfds_dataset=False):
    # If tfds_dataset is True, then name_or_path is the name of a dataset in the TensorFlow Datasets library
    # Otherwise, name_or_path is the path to a directory containing a dataset
    if tfds_dataset:
        return tfds.load(name_or_path, shuffle_files=True)

print(load_dataset('mnist', tfds_dataset=True))
