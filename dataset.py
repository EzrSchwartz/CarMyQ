import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from resnetclassification import ResNet152,ResNet101,ResNet50

import tqdm
from tqdm import tqdm
class PriusData(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Initializes the dataset.
        Args:
            annotations_file (str): Path to the CSV file with annotations (image_path, label).
            img_dir (str): Directory containing all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))
    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, label)
        """
        entries = os.listdir(self.img_dir)

        
        img_path = os.path.join(self.img_dir, entries[idx])
        image = Image.open(img_path).convert("RGB") # Open image and ensure RGB format
        label = "prius"

        if self.transform:
            image = self.transform(image)

        return image, label


class ConnieData(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Initializes the dataset.
        Args:
            annotations_file (str): Path to the CSV file with annotations (image_path, label).
            img_dir (str): Directory containing all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform


    def __len__(self):
        return len(os.listdir(self.img_dir))
    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, label)
        """
        entries = os.listdir(self.img_dir)

        
        img_path = os.path.join(self.img_dir, entries[idx])
        image = Image.open(img_path).convert("RGB") # Open image and ensure RGB format
        label = "Connie"

        if self.transform:
            image = self.transform(image)

        return image, label

class E63Data(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Initializes the dataset.
        Args:
            annotations_file (str): Path to the CSV file with annotations (image_path, label).
            img_dir (str): Directory containing all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform


    def __len__(self):
        return len(os.listdir(self.img_dir))
    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, label)
        """
        entries = os.listdir(self.img_dir)

        
        img_path = os.path.join(self.img_dir, entries[idx])   
        image = Image.open(img_path).convert("RGB") # Open image and ensure RGB format
        label = "e63"

        if self.transform:
            image = self.transform(image)

        return image, label


class OtherData(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Initializes the dataset.
        Args:
            annotations_file (str): Path to the CSV file with annotations (image_path, label).
            img_dir (str): Directory containing all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))
    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, label)
        """
        entries = os.listdir(self.img_dir)

        
        img_path = os.path.join(self.img_dir, entries[idx])
        image = Image.open(img_path).convert("RGB") # Open image and ensure RGB format
        label = "other"

        if self.transform:
            image = self.transform(image)

        return image, label


priusdataset = PriusData(img_dir=r"/home/ec2-user/VehiclesData/Prius/frames", transform=transforms.ToTensor())
# conniedataset = ConnieData(img_dir=r"D:\VehiclesData\Connie\frames", transform=transforms.ToTensor())
# e63dataset = E63Data(img_dir=r"D:\VehiclesData\E63\frames", transform=transforms.ToTensor())
otherdataset = OtherData(img_dir=r"/home/ec2-user/VehiclesData/Other/frames", transform=transforms.ToTensor())

# combined_dataset = torch.utils.data.ConcatDataset([priusdataset, conniedataset, e63dataset, otherdataset])
combined_dataset = torch.utils.data.ConcatDataset([priusdataset, otherdataset])


dataset = torch.utils.data.DataLoader(combined_dataset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_function = torch.nn.CrossEntropyLoss()
for model in [ResNet50(num_classes=4), ResNet101(num_classes=4), ResNet152(num_classes=4)]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Add model parameters later
    model.to(device)
    for epoch in tqdm(range(10)):
        for images, labels in tqdm(dataset):
            images = images.to(device)

            # Convert string labels to integers (you’ll need a mapping)
            label_map = {"prius": 0, "Connie": 1, "e63": 2, "other": 3}
            targets = torch.tensor([label_map[l] for l in labels], dtype=torch.long).to(device)

            output = model(images)
            loss = loss_function(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
        model.save_state_dict(torch.save(model.state_dict(), f"{model.__class__.__name__}{epoch}_model.pth"))


