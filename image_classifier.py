import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim

import torch
import torch.nn as nn

from image_cnn import net

from tqdm import tqdm

# Path to your dataset
data_dir = "new_images"

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),   # resize to consistent shape
    transforms.ToTensor(),         # convert to tensor (C x H x W) range [0,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1,1]
])

# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split into train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check the mapping of class names to numeric labels
print("Class to index mapping:", dataset.class_to_idx)
print("Num Classes:", len(dataset.class_to_idx.keys()))

class_names = list(dataset.class_to_idx.keys())
num_classes = len(class_names)

# Create a dictionary to store samples
samples_per_class = {cls: [] for cls in class_names}

# Collect up to 2 samples from each class
for img, label in dataset:
    class_name = class_names[label]
    if len(samples_per_class[class_name]) < 2:
        samples_per_class[class_name].append(img)
    if all(len(v) == 2 for v in samples_per_class.values()):
        break  # stop once we have 2 per class

# Plot them
fig, axes = plt.subplots(num_classes, 2, figsize=(6, 3 * num_classes))
if num_classes == 1:
    axes = [axes]  # handle single-class edge case

for i, cls in enumerate(class_names):
    for j, img_tensor in enumerate(samples_per_class[cls]):
        # Unnormalize (convert from [-1,1] back to [0,1])
        img = img_tensor * 0.5 + 0.5
        img = img.permute(1, 2, 0)  # CHW → HWC for plotting
        axes[i][j].imshow(img)
        axes[i][j].set_title(f"{cls} ({j+1})")
        axes[i][j].axis("off")

plt.tight_layout()
plt.show()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(enumerate(train_loader, 0), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    for i, data in train_bar:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{100 * correct / total:.2f}%"
        })

print('Finished Training')

dataiter = iter(test_loader)
images, labels = next(dataiter)

display_images = images * 0.5 + 0.5  # because normalized earlier to [-1,1]

# Make a grid for visualization
grid = torchvision.utils.make_grid(display_images)

# Convert from (C, H, W) to (H, W, C)
plt.imshow(grid.permute(1, 2, 0))

classes = list(dataset.class_to_idx.keys())

print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

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

print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

plt.show()