import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image



# 1a. Define how to transform each image before feeding to the model
#    - Resize to 224×224 (what ResNet expects)
#    - Turn into a tensor of numbers
#    - Normalize channels using ImageNet statistics
transform = transforms.Compose([
    transforms.Resize((224, 224)),            # make every image 224×224 pixels
    transforms.ToTensor(),                    # convert image to a PyTorch tensor
    transforms.Normalize(                     # shift and scale colors
        mean=[0.485, 0.456, 0.406],           # average RGB values in ImageNet
        std=[0.229, 0.224, 0.225]             # how much each channel varies
    )
])

# 1b. Load images from folders. Each subfolder name is treated as a class label.
dataset = datasets.ImageFolder("PlantVillage", transform=transform)

# 1c. Split into training (80%) and validation (20%)
train_size = int(0.8 * len(dataset))         # count of training examples
val_size   = len(dataset) - train_size       # count of validation examples
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 1d. Create PyTorch DataLoaders to handle batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)

num_classes = len(dataset.classes)          # how many leaf folders = number of diseases

# 2. LOAD A PRE-TRAINED MODEL (BACKBONE)
# --------------------------------------

# 2a. models.resnet18 comes ready-trained on ImageNet
model = models.resnet18(pretrained=True)

# 2b. “Head” is the final layer that turns features into class scores.
#    We replace it so it outputs `num_classes` instead of 1000.
in_features = model.fc.in_features        # how many features the old head received
model.fc = nn.Linear(in_features, num_classes)

# 2c. Freeze all layers except the new head
#    Freeze = “lock” weight updates so only the head learns at first
for param in model.parameters():
    param.requires_grad = False            # stop gradient updates everywhere
for param in model.fc.parameters():
    param.requires_grad = True             # allow updates in the head only

# 3. SET UP TRAINING DETAILS
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)                  # move model to GPU (if available)

criterion = nn.CrossEntropyLoss()         # loss for multi-class classification
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),  # only head params
    lr=0.001
)

# 4. TRAIN HEAD ONLY
# ------------------

epochs_head = 5
for epoch in range(epochs_head):
    model.train()                         # set to training mode
    running_loss, running_correct = 0.0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)           # forward pass
        loss = criterion(outputs, labels) # compute loss

        optimizer.zero_grad()             # clear old gradients
        loss.backward()                   # compute new gradients
        optimizer.step()                  # update head weights

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()

    epoch_loss = running_loss / train_size
    epoch_acc  = running_correct / train_size
    print(f"[Head Epoch {epoch+1}/{epochs_head}] "
          f"Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

    # (you can add a validation loop here if you like)



# If you have enough data and compute, you can now let the backbone refine itself
for param in model.parameters():
    param.requires_grad = True             # allow all layers to learn

# Use a smaller learning rate when fine-tuning everywhere
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs_finetune = 5
for epoch in range(epochs_finetune):
    model.train()
    running_loss, running_correct = 0.0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()

    epoch_loss = running_loss / train_size
    epoch_acc  = running_correct / train_size
    print(f"[Fine-tune Epoch {epoch+1}/{epochs_finetune}] "
          f"Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

    # Validate after each epoch
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
    val_loss /= val_size
    val_acc  = val_correct / val_size
    print(f"  → Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")
    model.train()

# 6. SAVE THE FINAL MODEL
# -----------------------

torch.save(model.state_dict(), "fine_tuned_resnet18.pth")  # keep weights for later

# 7. INFERENCE FUNCTION
# ---------------------

# Load a fresh model for inference
inference_model = models.resnet18(pretrained=False)
inference_model.fc = nn.Linear(in_features, num_classes)
inference_model.load_state_dict(torch.load("fine_tuned_resnet18.pth", map_location=device))
inference_model = inference_model.to(device)
inference_model.eval()                      # turn off dropout/batchnorm updates

def predict(image_path: str) -> str:
    # 1. Load the image in RGB
    img = Image.open(image_path).convert("RGB")
    
    # 2. Preprocess → returns a torch.Tensor
    tensor = transform(img)
    
    # 3. Add a batch dimension at dim=0 and move to the right device
    assert isinstance(tensor, torch.Tensor), f"Expected tensor, got {type(tensor)}"
    tensor = tensor.unsqueeze(0).to(device)  # use tensor.unsqueeze, not torch.unsqueeze
    
    # 4. Inference
    inference_model.eval()
    with torch.no_grad():
        logits = inference_model(tensor)
    
    # 5. Pick the most likely class
    idx = logits.argmax(dim=1).item()
    return dataset.classes[idx]

# Example:
# print(predict("some_leaf_image.jpg"))
