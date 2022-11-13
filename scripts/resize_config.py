def get_resize_config():
    return {
        "input_path": "/home/chen/big_bertha/coding/capstone/art-generator/experiments/data/original_full_sized_scapes",
        "output_path": "/home/chen/big_bertha/coding/capstone/art-generator/experiments/data/scapes_640_320_cropped_zero_pad_limit_0_5",
        "width": 640,
        "height": 360,
        "resize_factor": 0.01,
        "zero_pad_limit": 0.5,
        "crop": True
    }
