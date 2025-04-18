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
