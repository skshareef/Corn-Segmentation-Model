import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 3, kernel_size=1)
    return model

def load_model(path):
    model = get_model()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model
