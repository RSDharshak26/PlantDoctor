import torch 
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split  # this is for randomly splitting data between training and evaluation 
import torch.nn.functional as F
import torch.nn as nn


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(               # Normalize color channels
        mean=[0.485, 0.456, 0.406],    # Mean for each RGB channel (ImageNet stats)
        std=[0.229, 0.224, 0.225]      # Standard deviation for each RGB channel
    )
])
dataset = ImageFolder("/content/PlantDoctor/PlantVillage", transform=transform)

##dataset = ImageFolder("C:\\Users\rsdha\\Documents\\GitHub\\PlantDoctor\\PlantVillage",transform = transform)



# NEW: split dataset into train and validation sets (80/20 split)
train_len = int(0.8 * len(dataset))           # NEW: number of training samples
val_len = len(dataset) - train_len            # NEW: number of validation samples
train_set, val_set = random_split(dataset, [train_len, val_len])  # NEW: perform the split



# NEW: create DataLoaders for training and validation sets
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)  # NEW
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=2) # NEW


num_classes = len(dataset.classes)


##### Below is the architechture 
class DiseaseDetector(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.BatchNorm2d = nn.BatchNorm2d(16)
        self.MaxPool2d = nn.MaxPool2d(2, 2)
        self.Dropout = nn.Dropout(0.2)
        # INSERTING FLATTEN AND FCN LAYERS
        self.Flatten = nn.Flatten()  # Flattens all dimensions except batch
        self.fc1 = nn.Linear(16 * 111 * 111, 128)  # 16 channels, 111x111 each after pooling
        self.fc2 = nn.Linear(128, num_classes)     # Output layer
    
    def forward(self, x):
        x = self.conv1(x) #A convolutional layer outputs a 3D tensor: (number of filters, height, width) for each image.


        x = self.BatchNorm2d(x)
        x = F.relu(x)
        x = self.MaxPool2d(x)
        x = self.Dropout(x)
        
        # FLATTEN + FCN LAYERS
        x = self.Flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Fully connected (FC) layers expect a 1D vector per image as input.
        
        
        return x
    
    
    
    
# Instantiate the model
model = DiseaseDetector()


# Define loss function and optimizer here
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) ## need to clarify what are model parameters
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
model = model.to(device)



#moving model to gpu/cpu
    
for epoch in range(epochs):
    model.train()
    train_loss, train_correct = 0.0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
    train_loss /= len(train_set)
    train_acc  = train_correct / len(train_set)
    
    
    # --- Validation ---
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
    val_loss /= len(val_set)
    val_acc  = val_correct / len(val_set)
    
    
    print(f"[Epoch {epoch+1}/{epochs}] "
          f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
          f"Val   loss: {val_loss:.4f} acc: {val_acc:.4f}")
    
    
# ----------------- 4. SAVE WEIGHTS -----------------
torch.save(model.state_dict(), "disease_detector_final.pth")     # for later inference    





# --------------- INFERENCE CODE (do not modify anything above) ---------------

from PIL import Image

# load the trained model
inference_model = DiseaseDetector()                          # new model instance
inference_model.load_state_dict(torch.load("disease_detector_final.pth", map_location=device))  # load weights
inference_model.to(device)                                   # move to device
inference_model.eval()                                       # set to eval mode

# Rename the transform to avoid shadowing
preprocess = transform

def predict(image_path: str) -> str:
    # 1. Load as PIL
    pil_img = Image.open(image_path).convert("RGB")
    
    # 2. Preprocess into a tensor
    input_tensor = preprocess(pil_img)             # now clearly a torch.Tensor
    
    # 3. Add batch dimension & move to device
    input_tensor = input_tensor.unsqueeze(0).to(device)  # type: ignore
    
    # 4. Inference
    inference_model.eval()
    with torch.no_grad():
        logits = inference_model(input_tensor)
    
    # 5. Pick the most likely class
    pred_idx = logits.argmax(dim=1).item()
    return dataset.classes[pred_idx]                       # map index to class name

# Example usage:
# result = predict("C:\\Users\\rsdha\\Documents\\GitHub\\PlantDoctor\\test.jpg")
# print(f"Predicted class: {result}")

