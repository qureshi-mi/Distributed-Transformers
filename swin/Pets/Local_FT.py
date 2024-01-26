import torch
import os
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import torch.optim as optim
import numpy as np
import math
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

# Set a seed for reproducibility
seed = 107
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Split the batch data into 4 sets of distinct classes
def split_data(data, labels):
    l1 = torch.where((labels < 5) & (labels >= 0))[0]
    l2 = torch.where((labels < 20) & (labels >= 5))[0]
    l3 = torch.where((labels < 30) & (labels >= 20))[0]
    l4 = torch.where((labels < 37) & (labels >= 30))[0]
    data1 = data[l1,...]
    data2 = data[l2,...]
    data3 = data[l3,...]
    data4 = data[l4,...]
    return data1, data2, data3, data4

# Split the batch labels into 4 sets of distinct classes
def split_labels(labels):
    l1 = torch.where((labels < 5) & (labels >= 0))[0]
    l2 = torch.where((labels < 20) & (labels >= 5))[0]
    l3 = torch.where((labels < 30) & (labels >= 20))[0]
    l4 = torch.where((labels < 37) & (labels >= 30))[0]
    data1 = labels[l1,...]
    data2 = labels[l2,...]
    data3 = labels[l3,...]
    data4 = labels[l4,...]
    return data1, data2, data3, data4

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define hyperparameters
num_classes = 37
batch_size = 128
num_epochs = 200
learning_rate = 0.001
N = 4

class PetDataset(Dataset):
    "Dataset to serve individual images to our model"
    
    def __init__(self, data, transforms=None):
        self.data = data
        self.len = len(data)
        self.transforms = transforms
    
    def __getitem__(self, index):
        img_path, label = self.data[index]
        with open(img_path, 'rb') as img_file:
            img = Image.open(img_file).convert('RGB')
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, label
    
    def __len__(self):
        return self.len

# Define data directory
data_dir = './data/pets/images/'

# Load and preprocess Pets dataset
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load images using glob
image_files = glob(os.path.join(data_dir, '*.jpg'))

# Create labels based on image filenames (class names as labels)
labels = [os.path.basename(image_path).split('_')[0] for image_path in image_files]

# Create a mapping from class names to class indices
class_names = sorted(set(labels))
class2idx = {class_name: idx for idx, class_name in enumerate(class_names)}

# Create labels as class indices
labels = [class2idx[class_name] for class_name in labels]

# Create dataset
data = [(image_path, label) for image_path, label in zip(image_files, labels)]
dataset = PetDataset(data, transforms=data_transforms)

# Split dataset into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for train and test
# batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

# Load pre-trained swin model and move to GPU
model1 = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True)
model2 = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True)
model3 = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True)
model4 = timm.create_model('swin_small_patch4_window7_224.ms_in22k', pretrained=True)

model1.head.fc = nn.Linear(model1.head.fc.in_features, num_classes)
model2.head.fc = nn.Linear(model2.head.fc.in_features, num_classes)
model3.head.fc = nn.Linear(model3.head.fc.in_features, num_classes)
model4.head.fc = nn.Linear(model4.head.fc.in_features, num_classes)

model1.to(device)
model2.to(device)
model3.to(device)
model4.to(device)

# Freezing the parameters of all blocks except the last one     
for (name1, param1), (name2, param2), (name3, param3), (name4, param4) in zip(model1.named_parameters(), model2.named_parameters(), model3.named_parameters(), model4.named_parameters()):
    if "layers" in name1 and not "layers.3.blocks.1" in name1:
        param1.requires_grad = False
    else:
        param1.requires_grad = True

    if "layers" in name2 and not "layers.3.blocks.1" in name2:
        param2.requires_grad = False
    else:
        param2.requires_grad = True

    if "layers" in name3 and not "layers.3.blocks.1" in name3:
        param3.requires_grad = False
    else:
        param3.requires_grad = True

    if "layers" in name4 and not "layers.3.blocks.1" in name4:
        param4.requires_grad = False
    else:
        param4.requires_grad = True

# Define loss and optimizer for only the last block
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model1.train()
    model2.train()
    model3.train()
    model4.train()
    
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    total_loss4 = 0.0

    correct_predictions1 = 0
    correct_predictions2 = 0
    correct_predictions3 = 0
    correct_predictions4 = 0

    num_samples1 = 0
    num_samples2 = 0
    num_samples3 = 0
    num_samples4 = 0
    
    i = 0
    for images, labels in train_loader:
        # splitting the data and labels
        d1, d2, d3, d4 = split_data(images, labels)
        l1, l2, l3, l4 = split_labels(labels)
        d1, d2, d3, d4 = d1.to(device), d2.to(device), d3.to(device), d4.to(device)  # Move data to GPU
        l1, l2, l3, l4 = l1.to(device), l2.to(device), l3.to(device), l4.to(device)  # Move data to GPU
        
        num_samples1 += len(d1)
        num_samples2 += len(d2)
        num_samples3 += len(d3)
        num_samples4 += len(d4)
        
        outputs1 = model1(d1)
        outputs2 = model2(d2)
        outputs3 = model3(d3)
        outputs4 = model4(d4)

        loss1 = criterion(outputs1, l1)
        loss2 = criterion(outputs2, l2)
        loss3 = criterion(outputs3, l3)
        loss4 = criterion(outputs4, l4)

        loss1.backward()
        loss2.backward()
        loss3.backward()
        loss4.backward()
        
        # Update the parameters of the last block
        with torch.no_grad():
            for (name1, param1), (name2, param2), (name3, param3), (name4, param4) in zip(model1.named_parameters(), model2.named_parameters(), model3.named_parameters(), model4.named_parameters()):
                
                if param1.requires_grad and param2.requires_grad and param3.requires_grad and param4.requires_grad:
                    param1 -= learning_rate * param1.grad
                    param1.grad.zero_()  # Reset gradient
                        
                    param2 -= learning_rate * param2.grad
                    param2.grad.zero_()  # Reset gradient
                        
                    param3 -= learning_rate * param3.grad
                    param3.grad.zero_()  # Reset gradient
                        
                    param4 -= learning_rate * param4.grad
                    param4.grad.zero_()  # Reset gradient

        i += 1
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_loss3 += loss3.item()
        total_loss4 += loss4.item()
        
        _, predicted1 = torch.max(outputs1, 1)
        _, predicted2 = torch.max(outputs2, 1)
        _, predicted3 = torch.max(outputs3, 1)
        _, predicted4 = torch.max(outputs4, 1)
        
        correct_predictions1 += (predicted1 == l1).sum().item()
        correct_predictions2 += (predicted2 == l2).sum().item()
        correct_predictions3 += (predicted3 == l3).sum().item()
        correct_predictions4 += (predicted4 == l4).sum().item()

        if i % 100 == 0:
            print("Iteration: " + str(i))
    
    accuracy1 = correct_predictions1 / num_samples1 * 100
    accuracy2 = correct_predictions2 / num_samples2 * 100
    accuracy3 = correct_predictions3 / num_samples3 * 100
    accuracy4 = correct_predictions4 / num_samples4 * 100
    
    epoch_loss1 = total_loss1 / num_samples1
    epoch_loss2 = total_loss2 / num_samples2
    epoch_loss3 = total_loss3 / num_samples3
    epoch_loss4 = total_loss4 / num_samples4
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss1:.4f}, Train Accuracy: {accuracy1:.2f}%')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss2:.4f}, Train Accuracy: {accuracy2:.2f}%')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss3:.4f}, Train Accuracy: {accuracy3:.2f}%')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss4:.4f}, Train Accuracy: {accuracy4:.2f}%')

    # Testing loop
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    test_correct_predictions1 = 0
    test_correct_predictions2 = 0
    test_correct_predictions3 = 0
    test_correct_predictions4 = 0
    
    # evaluate the test accuracy
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs1 = model1(images)
            outputs2 = model2(images)
            outputs3 = model3(images)
            outputs4 = model4(images)
            
            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)
            _, predicted3 = torch.max(outputs3, 1)
            _, predicted4 = torch.max(outputs4, 1)
            
            test_correct_predictions1 += (predicted1 == labels).sum().item()
            test_correct_predictions2 += (predicted2 == labels).sum().item()
            test_correct_predictions3 += (predicted3 == labels).sum().item()
            test_correct_predictions4 += (predicted4 == labels).sum().item()
    
    test_accuracy1 = test_correct_predictions1 / len(test_dataset) * 100
    test_accuracy2 = test_correct_predictions2 / len(test_dataset) * 100
    test_accuracy3 = test_correct_predictions3 / len(test_dataset) * 100
    test_accuracy4 = test_correct_predictions4 / len(test_dataset) * 100

    print(f'Test Accuracy: {test_accuracy1:.2f}%')
    print(f'Test Accuracy: {test_accuracy2:.2f}%')
    print(f'Test Accuracy: {test_accuracy3:.2f}%')
    print(f'Test Accuracy: {test_accuracy4:.2f}%')
    
    # Save the test accuracy of each node
    file = open('results/LocalFT_test_acc1.txt', 'a')
    file.write("%s\n" % float(test_accuracy1))
    file.close()
    file = open('results/LocalFT_test_acc2.txt', 'a')
    file.write("%s\n" % float(test_accuracy2))
    file.close()
    file = open('results/LocalFT_test_acc3.txt', 'a')
    file.write("%s\n" % float(test_accuracy3))
    file.close()
    file = open('results/LocalFT_test_acc4.txt', 'a')
    file.write("%s\n" % float(test_accuracy4))
    file.close()

print('Training finished!')

