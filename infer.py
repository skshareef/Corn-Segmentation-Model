### infer.py
from utils.model import load_model
from utils.helpers import predict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, help="Path to input image")
    args = parser.parse_args()

    model = load_model("outputs/checkpoints/best_model.pt")
    predict(model, args.img)


### config.py
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
