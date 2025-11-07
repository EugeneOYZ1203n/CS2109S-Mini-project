import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim

import torch
import torch.nn as nn

from image_cnn import Net

from tqdm import tqdm

from utils import generate_torch_loader_snippet

# Path to your dataset
data_dir = "new_images"

# Define image transformations
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 4. Create wrapper classes that apply different transforms
class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir)

# Split into train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(TransformedSubset(train_dataset, train_transform), batch_size=32, shuffle=True)
test_loader = DataLoader(TransformedSubset(test_dataset, test_transform), batch_size=32, shuffle=False)

# Check the mapping of class names to numeric labels
print("Class to index mapping:", dataset.class_to_idx)
print("Num Classes:", len(dataset.class_to_idx.keys()))

class_names = list(dataset.class_to_idx.keys())
num_classes = len(class_names)

net = Net(num_classes)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 50
start_epoch = 0

is_load = True
is_save_snippet = False
is_save_after_snippet = True
if is_load:
    checkpoint_path = "checkpoints/cnn_epoch_50.pth" 

    checkpoint = torch.load(checkpoint_path)

    # Recreate model with correct class count
    net = Net(checkpoint['num_classes'])
    net.load_state_dict(checkpoint['model_state_dict'])

    # Recreate optimizer and scheduler
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")

    if is_save_snippet:
        loader_snippet = generate_torch_loader_snippet(net)
                
        output_path = "img_model_loader_snippet.txt"

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(loader_snippet)

        print(f"✅ Loader snippet saved to {output_path}")

epoch_bar = tqdm(range(start_epoch, start_epoch + num_epochs), desc="Epochs", position=0)
for epoch in epoch_bar:
    net.train(True)
    
    running_loss = 0.

    _correct = 0
    _total = 0

    train_bar = tqdm(enumerate(train_loader, 0), desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}", leave=False, total=len(train_loader), position=1)
    for i, data in train_bar:
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        _total += labels.size(0)
        _correct += (predicted == labels).sum().item()

        train_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{100 * _correct / _total:.2f}%",
            "LR": optimizer.param_groups[0]['lr']
        })
    
    scheduler.step()

    net.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_bar.set_postfix({
        "Test ACC": f"{correct/total:%}",
    })

save_path = "checkpoints"
os.makedirs(save_path, exist_ok=True)

checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': running_loss,
    'accuracy': 100 * _correct / _total,
    'num_classes': num_classes
}

torch.save(checkpoint, os.path.join(save_path, f'cnn_epoch_{epoch+1}.pth'))
print(f"Checkpoint saved: cnn_epoch_{epoch+1}.pth")

if is_save_after_snippet:
    loader_snippet = generate_torch_loader_snippet(net)
            
    output_path = "img_model_loader_snippet.txt"

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(loader_snippet)

    print(f"✅ Loader snippet saved to {output_path}")