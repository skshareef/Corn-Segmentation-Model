
TRAIN_CONFIG = {
    "data": {
        "img_dir": "data/images",
        "msk_dir": "data/masks",
        "batch_size": 32,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "num_workers": 4,
        "seed": 42
    },
    "training": {
        "epochs": 50,
        "lr": 1e-4,
        "checkpoint_path": "outputs/checkpoints/",
        "log_path": "outputs/logs/"
    }
}
