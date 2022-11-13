import os, sys
import cv2
from concurrent.futures import ThreadPoolExecutor
from resize_config import get_resize_config


def scale_image(image, width, height, rescale_factor):
    if image_fits_in_desired_size(image, width, height):
        # Grow image
        if image.shape[0] > image.shape[1]:
            while image.shape[0] < height:
                image = cv2.resize(
                    image, (0, 0), fx=1 + rescale_factor, fy=1 + rescale_factor
                )
        else:
            while image.shape[1] < width:
                image = cv2.resize(
                    image, (0, 0), fx=1 + rescale_factor, fy=1 + rescale_factor
                )

    else:
        # Shrink image
        if image.shape[0] > image.shape[1]:
            while image.shape[0] > height:
                image = cv2.resize(
                    image, (0, 0), fx=1 - rescale_factor, fy=1 - rescale_factor
                )
        else:
            while image.shape[1] > width:
                image = cv2.resize(
                    image, (0, 0), fx=1 - rescale_factor, fy=1 - rescale_factor
                )

        # Reverse the last resize
        image = cv2.resize(image, (0, 0), fx=1 + rescale_factor, fy=1 + rescale_factor)

    return image

def image_fits_in_desired_size(image, width, height):
    return image.shape[0] <= height and image.shape[1] <= width


def resize_image(
    image_path, output_path, width, height, rescale_factor=0.1, zero_pad_limit=0.5
):
    image = cv2.imread(image_path)

    if not config["crop"]:
        image = scale_image(image, width, height, rescale_factor)


    # Crop image from center
    if config["crop"] and image.shape[0] > height:
        start_height = int((image.shape[0] - height) / 2)
        end_height = start_height + height
        image = image[start_height:end_height, :]

    if config["crop"] and image.shape[1] > width:
        start_width = int((image.shape[1] - width) / 2)
        end_width = start_width + width
        image = image[:, start_width:end_width]


    # Zero pad image
    if image.shape[0] < height or image.shape[1] < width:
        if (
            1 - (image.shape[0] / height) > zero_pad_limit
            or 1 - (image.shape[1] / width) > zero_pad_limit
        ):
            # Skip image which is going to be padded more than the limit
            return

        image = cv2.copyMakeBorder(
            image,
            0,
            height - image.shape[0],
            0,
            width - image.shape[1],
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

    cv2.imwrite(output_path, image)


def prepare_output_dirs(config):
    if not os.path.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    for category in os.listdir(config["input_path"]):
        if category == "attributions.json":
            continue
        category_dir = os.path.join(config["output_path"], category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)


def get_resize_image_args(config):
    args = []
    for category in os.listdir(config["input_path"]):
        if category == "attributions.json":
            continue
        category_dir = os.path.join(config["input_path"], category)
        for image in os.listdir(category_dir):
            if image == "attributions.json":
                continue
            image_path = os.path.join(category_dir, image)
            # Save images as PNG for lossless compression
            image = ".".join(image.split(".")[:-1]) + ".png"
            output_image_path = os.path.join(config["output_path"], category, image)
            args.append(
                (
                    image_path,
                    output_image_path,
                    config["width"],
                    config["height"],
                    config["resize_factor"],
                    config["zero_pad_limit"],
                )
            )
    return args


# Verify that the dimensions of the images in the resized dataset are correct

invalid_image_count = 0


def check_image_size(image_path, width, height):
    global invalid_image_count
    image = cv2.imread(image_path)
    if image.shape[0] != height or image.shape[1] != width:
        # print('Invalid image: {}'.format(image_path))
        invalid_image_count += 1


if __name__ == "__main__":
    config = get_resize_config()
    # prepare_output_dirs(config)
    # resize_image_args = get_resize_image_args(config)
    # with ThreadPoolExecutor() as executor:
    #     # We unpack the arguments here because the zip function expects a comma separated list of arguments
    #     # We unpack the result of the zip function because the map function expects a comma separated list of arguments
    #     # e.g. map(func, (arg1, arg2), (arg1, arg2)) and zip((arg1, arg1), (arg2, arg2))
    #     _ = executor.map(resize_image, *zip(*resize_image_args))

    check_image_size_args = []
    for category in os.listdir(config["output_path"]):
        category_dir = os.path.join(config["output_path"], category)
        for image in os.listdir(category_dir):
            check_image_size_args.append(
                (os.path.join(category_dir, image), config["width"], config["height"])
            )

    with ThreadPoolExecutor() as executor:
        executor.map(check_image_size, *zip(*check_image_size_args))

    print("Found {} invalid images".format(invalid_image_count))
