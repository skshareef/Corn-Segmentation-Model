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


