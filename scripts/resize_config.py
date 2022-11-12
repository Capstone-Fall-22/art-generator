def get_resize_config():
    return {
        "input_path": "./experiments/data/full_sized_dataset",
        "output_path": "./experiments/data/scapes_resized_cleaned",
        "width": 1280,
        "height": 720,
        "resize_factor": 0.01,
        "zero_pad_limit": 0.1,
    }
