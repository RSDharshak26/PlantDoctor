import torch 
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
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


dataset = ImageFolder("C:\\Users\rsdha\\Documents\\GitHub\\PlantDoctor\\PlantVillage",transform = transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

##### Below is the architechture 
class DiseaseDetector(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.BatchNorm2d = nn.BatchNorm2d(16)
        self.MaxPool2d = nn.MaxPool2d(2, 2)
        self.Dropout = nn.Dropout(0.2)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.BatchNorm2d(x)
        x = F.relu(x)
        x = self.MaxPool2d(x)
        x = self.Dropout(x)
        # Continue forward pass
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
    running_loss, running_correct = 0.0, 0 ## need to understand this 
    
    for images,labels in dataloader:
    #1. getting the predictions 
        outputs = model(images)
        loss = criterion(outputs,labels)
        
        optimizer.zero_grad()                                   # 3a. clear old grads
        loss.backward()                                         # 3b. back-prop: compute grads
        optimizer.step()
        
         # ---------- metrics ----------
        running_loss += loss.item() * images.size(0)            # sum batch loss × batch_size
        preds = outputs.argmax(dim=1)                           # predicted class per sample
        running_correct += (preds == labels).sum().item()       # correct predictions count

    # -------- epoch summary --------
    epoch_loss = running_loss / len(dataset)
    epoch_acc  = running_correct / len(dataset)
    print(f"[Epoch {epoch+1}/{epochs}]  loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")
    
    
# ----------------- 4. SAVE WEIGHTS -----------------
torch.save(model.state_dict(), "disease_detector_final.pth")     # for later inference    