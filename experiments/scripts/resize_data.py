import os, sys
import cv2
from concurrent.futures import ThreadPoolExecutor

def imageFitsInDesiredSize(image, width, height):
    return image.shape[0] <= height and image.shape[1] <= width

def resize_image(image_path, output_path, width, height, rescale_factor=0.1):
    # print('Resizing image: {}'.format(image_path))
    image = cv2.imread(image_path)
    if imageFitsInDesiredSize(image, width, height):
        # Grow image
        if image.shape[0] > image.shape[1]:
            while image.shape[0] < height:
                image = cv2.resize(image, (0,0), fx=1+rescale_factor, fy=1+rescale_factor)
        else:
            while image.shape[1] < width:
                image = cv2.resize(image, (0,0), fx=1+rescale_factor, fy=1+rescale_factor)

    else:
        # Shrink image
        if image.shape[0] > image.shape[1]:
            while image.shape[0] > height:
                image = cv2.resize(image, (0,0), fx=1-rescale_factor, fy=1-rescale_factor)
        else:
            while image.shape[1] > width:
                image = cv2.resize(image, (0,0), fx=1-rescale_factor, fy=1-rescale_factor)
        
        # Reverse the last resize
        image = cv2.resize(image, (0,0), fx=1+rescale_factor, fy=1+rescale_factor)

    # Crop image
    image = image[0:height, 0:width]

    # Zero pad image
    if image.shape[0] < height or image.shape[1] < width:
        # Add 0 from top, desirsed_height - actual_height for bottom, 0 for left, desired_width - actual_width for right
        image = cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, width - image.shape[1], cv2.BORDER_CONSTANT, value=[0,0,0]) 

    cv2.imwrite(output_path, image)

def prepare_output_dirs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for category in os.listdir(input_path):
        if category == 'attributions.json':
            continue
        category_dir = os.path.join(output_path, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)

def get_resize_image_args(input_path, output_path, width, height, resize_factor):
    args = []
    for category in os.listdir(input_path):
        if category == 'attributions.json':
            continue
        category_dir = os.path.join(input_path, category)
        for image in os.listdir(category_dir):
            if image == 'attributions.json':
                continue
            image_path = os.path.join(category_dir, image)
            output_image_path = os.path.join(output_path, category, image)
            args.append((image_path, output_image_path, width, height, resize_factor))
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
    input_path, output_path, width, height, rescale_factor = sys.argv[1:]
    width = int(width)
    height = int(height)
    rescale_factor = float(rescale_factor)
    prepare_output_dirs(input_path, output_path)
    resize_image_args = get_resize_image_args(input_path, output_path, width, height, rescale_factor)
    with ThreadPoolExecutor() as executor:
        # We unpack the arguments here because the zip function expects a comma separated list of arguments
        # We unpack the result of the zip function because the map function expects a comma separated list of arguments
        # e.g. map(func, (arg1, arg2), (arg1, arg2)) and zip((arg1, arg1), (arg2, arg2))
        _ = executor.map(resize_image, *zip(*resize_image_args))

    width = int(width)
    height = int(height)
    check_image_size_args = []
    for category in os.listdir(output_path):
        category_dir = os.path.join(output_path, category)
        for image in os.listdir(category_dir):
            check_image_size_args.append((os.path.join(category_dir, image), width, height))

    with ThreadPoolExecutor() as executor:
        executor.map(check_image_size, *zip(*check_image_size_args))
        
    print('Found {} invalid images'.format(invalid_image_count))