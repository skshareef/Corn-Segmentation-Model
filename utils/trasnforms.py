from torchvision import transforms as T
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, HueSaturationValue, RandomBrightnessContrast, CLAHE, RandomGamma, GaussianBlur
from albumentations.pytorch import ToTensorV2

def build_img_transform(size=512):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def build_mask_transform(size=512):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor()
    ])

def build_train_aug():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        RandomBrightnessContrast(p=0.5),
        CLAHE(p=0.5),
        RandomGamma(p=0.5),
        GaussianBlur(p=0.3)
    ])
