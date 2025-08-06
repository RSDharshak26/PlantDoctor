from flask import Blueprint, request , jsonify
inference = Blueprint('inference',__name__)

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Add this transform definition (same as your training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),            # make every image 224×224 pixels
    transforms.ToTensor(),                    # convert image to a PyTorch tensor
    transforms.Normalize(                     # shift and scale colors
        mean=[0.485, 0.456, 0.406],           # average RGB values in ImageNet
        std=[0.229, 0.224, 0.225]             # how much each channel varies
    )
])


# step 1 : load pre trained model weights
def load_model():
    global model, class_names
    
    # Create the same model architecture
    model = models.resnet18(pretrained=False)
    #This line creates a ResNet18 model with random weights (not pre-trained).
    #It gives you the architecture (the layers and structure) of ResNet18.
     
    model.fc = nn.Linear(model.fc.in_features, 38)  # Replace 38 with your num_classes    
    #     model.fc is the final fully connected (FC) layer of ResNet18.
    # .in_features is the number of input features to the FC layer (depends on ResNet architecture, usually 512).
    # 38 is the number of classes you want to predict (change this to match your dataset).
    # This line replaces the original FC layer (which outputs 1000 classes for ImageNet) with a new one for your task.   
    
    # Load your trained weights
    model.load_state_dict(torch.load("fine_tuned_resnet18.pth", map_location="cpu"))
    # model.load_state_dict(...) fills in all the weights for every layer in your model (including the new FC layer, if you trained it).
 
    model.eval()
    
    # Define your class names (replace with your actual disease names)
    class_names =['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


# step 2 : preprocess uploaded image
def preprocess_image(file):
    # Read image from uploaded file
    image = Image.open(file.stream).convert("RGB")
    
    # Apply transforms
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor



# step 3 : run inference
def run_inference(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_idx = outputs.argmax(dim=1).item()
        predicted_class = class_names[predicted_idx]
        confidence = torch.softmax(outputs, dim=1).max().item()
    
    return predicted_class, confidence



# step 4 : updated route
# Load model when server starts
load_model()

@inference.route("/inference", methods=["GET", "POST", "OPTIONS"])
def process():
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        from flask import make_response
        resp = make_response('')
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp
    
    # Handle GET request
    if request.method == 'GET':
        from flask import make_response
        response = {"message": "Plant Doctor API", "status": "ready", "endpoint": "/inference"}
        resp = make_response(response)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp
    
    # Handle POST request
    try:
        # Get uploaded file
        file = request.files.get("file")
        if not file:
            from flask import make_response
            resp = make_response({"error": "No file uploaded", "status": "error"})
            resp.headers['Access-Control-Allow-Origin'] = '*'
            resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return resp
        print("file received")
        
        # Preprocess image
        image_tensor = preprocess_image(file)
        
        # Run inference
        predicted_class, confidence = run_inference(image_tensor)
        print("inference done")
        print("predicted_class : ", predicted_class)
        
        # Return results
        from flask import make_response
        response = {
            "prediction": predicted_class,
            "confidence": round(confidence, 3),
            "status": "success"
        }
        resp = make_response(response)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp
        
    except Exception as e:
        from flask import make_response
        resp = make_response({"error": str(e), "status": "error"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp


# Lambda entry point
def lambda_handler(event, context):
    # This will be your Lambda URL endpoint
    return process()