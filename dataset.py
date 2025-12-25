# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# import os
# import pandas as pd
# from resnetclassification import ResNet152,ResNet101,ResNet50

# import tqdm
# from tqdm import tqdm
# class PriusData(Dataset):
#     def __init__(self, img_dir, transform=None):
#         """
#         Initializes the dataset.
#         Args:
#             annotations_file (str): Path to the CSV file with annotations (image_path, label).
#             img_dir (str): Directory containing all the images.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return len(os.listdir(self.img_dir))
#     def __getitem__(self, idx):
#         """
#         Loads and returns a sample from the dataset at the given index.
#         Args:
#             idx (int): Index of the sample to retrieve.
#         Returns:
#             tuple: (image, label)
#         """
#         entries = os.listdir(self.img_dir)

        
#         img_path = os.path.join(self.img_dir, entries[idx])
#         image = Image.open(img_path).convert("RGB") # Open image and ensure RGB format
#         label = "prius"

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# class ConnieData(Dataset):
#     def __init__(self, img_dir, transform=None):
#         """
#         Initializes the dataset.
#         Args:
#             annotations_file (str): Path to the CSV file with annotations (image_path, label).
#             img_dir (str): Directory containing all the images.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.img_dir = img_dir
#         self.transform = transform


#     def __len__(self):
#         return len(os.listdir(self.img_dir))
#     def __getitem__(self, idx):
#         """
#         Loads and returns a sample from the dataset at the given index.
#         Args:
#             idx (int): Index of the sample to retrieve.
#         Returns:
#             tuple: (image, label)
#         """
#         entries = os.listdir(self.img_dir)

        
#         img_path = os.path.join(self.img_dir, entries[idx])
#         image = Image.open(img_path).convert("RGB") # Open image and ensure RGB format
#         label = "Connie"

#         if self.transform:
#             image = self.transform(image)

#         return image, label

# class E63Data(Dataset):
#     def __init__(self, img_dir, transform=None):
#         """
#         Initializes the dataset.
#         Args:
#             annotations_file (str): Path to the CSV file with annotations (image_path, label).
#             img_dir (str): Directory containing all the images.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.img_dir = img_dir
#         self.transform = transform


#     def __len__(self):
#         return len(os.listdir(self.img_dir))
#     def __getitem__(self, idx):
#         """
#         Loads and returns a sample from the dataset at the given index.
#         Args:
#             idx (int): Index of the sample to retrieve.
#         Returns:
#             tuple: (image, label)
#         """
#         entries = os.listdir(self.img_dir)

        
#         img_path = os.path.join(self.img_dir, entries[idx])   
#         image = Image.open(img_path).convert("RGB") # Open image and ensure RGB format
#         label = "e63"

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# class OtherData(Dataset):
#     def __init__(self, img_dir, transform=None):
#         """
#         Initializes the dataset.
#         Args:
#             annotations_file (str): Path to the CSV file with annotations (image_path, label).
#             img_dir (str): Directory containing all the images.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return len(os.listdir(self.img_dir))
#     def __getitem__(self, idx):
#         """
#         Loads and returns a sample from the dataset at the given index.
#         Args:
#             idx (int): Index of the sample to retrieve.
#         Returns:
#             tuple: (image, label)
#         """
#         entries = os.listdir(self.img_dir)

        
#         img_path = os.path.join(self.img_dir, entries[idx])
#         image = Image.open(img_path).convert("RGB") # Open image and ensure RGB format
#         label = "other"

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# priusdataset = PriusData(img_dir=r"/home/ec2-user/VehiclesData/Prius/frames", transform=transforms.ToTensor())
# # conniedataset = ConnieData(img_dir=r"D:\VehiclesData\Connie\frames", transform=transforms.ToTensor())
# # e63dataset = E63Data(img_dir=r"D:\VehiclesData\E63\frames", transform=transforms.ToTensor())
# otherdataset = OtherData(img_dir=r"/home/ec2-user/VehiclesData/Other/frames", transform=transforms.ToTensor())

# # combined_dataset = torch.utils.data.ConcatDataset([priusdataset, conniedataset, e63dataset, otherdataset])
# combined_dataset = torch.utils.data.ConcatDataset([priusdataset, otherdataset])


# dataset = torch.utils.data.DataLoader(combined_dataset, batch_size=1, shuffle=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loss_function = torch.nn.CrossEntropyLoss()
# for model in [ResNet50(num_classes=4), ResNet101(num_classes=4), ResNet152(num_classes=4)]:
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Add model parameters later
#     model.to(device)
#     for epoch in tqdm(range(10)):
#         for images, labels in tqdm(dataset):
#             images = images.to(device)

#             # Convert string labels to integers (you’ll need a mapping)
#             label_map = {"prius": 0, "Connie": 1, "e63": 2, "other": 3}
#             targets = torch.tensor([label_map[l] for l in labels], dtype=torch.long).to(device)

#             output = model(images)
#             loss = loss_function(output, targets)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         print(f"Epoch {epoch}, Loss: {loss.item()}")
#         model.save_state_dict(torch.save(model.state_dict(), f"{model.__class__.__name__}{epoch}_model.pth"))


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
import tqdm
from tqdm import tqdm
from resnetclassification import ResNet50  # Just use one model


class CarDataset(Dataset):
    def __init__(self, img_dir, is_prius, transform=None):
        self.img_dir = img_dir
        self.label = 1 if is_prius else 0  # 1 = prius, 0 = not prius
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# # Datasets
# prius_dataset = CarDataset(r"D:\VehiclesData\Prius\frames", is_prius=True, transform=transform)
# other_dataset = CarDataset(r"D:\VehiclesData\Other\frames", is_prius=False, transform=transform)

# combined = ConcatDataset([prius_dataset, other_dataset])
# dataloader = DataLoader(combined, batch_size=32, shuffle=True)

# print(f"Prius images: {len(prius_dataset)}, Other images: {len(other_dataset)}")

# # Model - binary classification (2 classes)
model = ResNet50(num_classes=2).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# loss_fn = nn.CrossEntropyLoss()

# Train
# model.train()

# for epoch in range(10):
#     total_loss, correct, total = 0, 0, 0
    
#     for images, labels in tqdm(dataloader):
#         images, labels = images.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
#         correct += (outputs.argmax(1) == labels).sum().item()
#         total += labels.size(0)
    
#     print(f"Epoch {epoch+1}/10 | Loss: {total_loss/len(dataloader):.4f} | Acc: {100*correct/total:.1f}%")

# torch.save(model.state_dict(), "prius_detector.pth")


# ============================================
# Inference - Get Prius Confidence
# ============================================

def get_prius_confidence(image_path, model, transform, device):
    """Returns confidence (0-1) that image contains a Prius"""
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        prius_confidence = probs[0, 1].item()  # Index 1 = prius
    
    return prius_confidence


# Usage
def getConfidence(image_path):
    model = ResNet50(num_classes=2).to(device)
    model.load_state_dict(torch.load("prius_detector.pth", map_location=device))
    confidence = get_prius_confidence(image_path, model, transform, device)
    return confidence


# model.load_state_dict(torch.load("prius_detector.pth"))

# confidence = get_prius_confidence(r"D:\VehiclesData\Prius\frames\frame_00091(night).png", model, transform, device)
# print(f"Prius confidence: {confidence:.1%}")  # e.g., "Prius confidence: 87.3%"