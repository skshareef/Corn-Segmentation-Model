import torch
import os
from tqdm import tqdm

def train_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(config["training"]["epochs"]):
        model.train()
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, masks = imgs.to(device), masks.squeeze(1).long().to(device)
            optimizer.zero_grad()
            output = model(imgs)['out']
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()

        # save model checkpoint
        ckpt_path = os.path.join(config["training"]["checkpoint_path"], f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)


def predict(model, image_path):
    from PIL import Image
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    import numpy as np

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        pred = torch.argmax(output, dim=0).numpy()

    plt.imshow(pred)
    plt.title("Predicted Mask")
    plt.show()
