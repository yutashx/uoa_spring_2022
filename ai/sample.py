from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Transform を作成する。
transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
# Dataset を作成する。
dataset = ImageFolder("../data/MNIST-JPG/JPGS/test/", transform)
print(dataset.class_to_idx)

# DataLoader を作成する。
dataloader = DataLoader(dataset, batch_size=3)

for X_batch, y_batch in dataloader:
    print(X_batch.shape, y_batch.shape)
