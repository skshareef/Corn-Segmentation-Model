from utils.dataset import build_loaders
from utils.model import get_model
from utils.helpers import train_model
from config import TRAIN_CONFIG

if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_loaders(**TRAIN_CONFIG["data"])
    model = get_model()
