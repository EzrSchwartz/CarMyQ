import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from resnetclassification import ResNet152,ResNet101,ResNet50
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


priusdataset = PriusData(img_dir='D:\VehiclesData\Prius\frames', transform=transforms.ToTensor())
conniedataset = ConnieData(img_dir='D:\VehiclesData\Connie\frames', transform=transforms.ToTensor())
e63dataset = E63Data(img_dir='D:\VehiclesData\E63\frames', transform=transforms.ToTensor())
otherdataset = OtherData(img_dir='D:\VehiclesData\Other\frames', transform=transforms.ToTensor())

combined_dataset = torch.utils.data.ConcatDataset([priusdataset, conniedataset, e63dataset, otherdataset])
dataset = torch.utils.data.DataLoader(combined_dataset, batch_size=32, shuffle=True)

for model in [ResNet50(num_classes=4), ResNet101(num_classes=4), ResNet152(num_classes=4)]:
    for epoch in range(10):
        for input_tensor in dataset:
            # Move input tensor to device
            input_tensor = input_tensor.to(device)

            # Forward pass
            output = model(input_tensor)

            # Compute loss, backpropagate, and update weights
            # (Assuming you have defined a loss function and optimizer)
            loss = loss_function(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

