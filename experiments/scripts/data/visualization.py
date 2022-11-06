import math
import matplotlib.pyplot as plt

def display_images(images, max_images=None):
    if max_images:
        images = images[:max_images]

    closest_square_root = math.ceil(math.sqrt(images.shape[0]))
    fig = plt.figure(figsize=(closest_square_root * 2, closest_square_root * 2))

    for i in range(len(images)):
        fig.add_subplot(closest_square_root, closest_square_root, i + 1)
        plt.imshow(images[i])


