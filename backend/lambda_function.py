import json
import base64
import boto3
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

# Global variables for model caching
model = None
class_names = None
transform = None

def load_model():
    """Load the PyTorch model from S3"""
    global model, class_names, transform
    
    if model is not None:
        return model, class_names, transform
    
    # Transform definition (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create model architecture
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 38)  # 38 classes
    
    # Load model weights from S3
    s3 = boto3.client('s3')
    bucket_name = os.environ.get('MODEL_BUCKET')
    model_key = os.environ.get('MODEL_KEY', 'fine_tuned_resnet18.pth')
    
    s3.download_file(bucket_name, model_key, '/tmp/model.pth')
    model.load_state_dict(torch.load('/tmp/model.pth', map_location='cpu'))
    model.eval()
    
    # Define class names
    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    
    return model, class_names, transform

def lambda_handler(event, context):
    """Main Lambda handler function"""
    try:
        # Load model (cached after first invocation)
        load_model()
        
        # Parse the incoming request
        body = json.loads(event.get('body', '{}'))
        
        # Get base64 image data
        image_data = body.get('file') or body.get('image')
        if not image_data:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': 'No file uploaded', 'status': 'error'})
            }
        
        # Decode and preprocess image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            outputs = model(tensor)
            predicted_idx = outputs.argmax(dim=1).item()
            predicted_class = class_names[predicted_idx]
            confidence = torch.softmax(outputs, dim=1).max().item()
        
        # Return results
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'prediction': predicted_class,
                'confidence': round(confidence, 3),
                'status': 'success'
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': str(e), 'status': 'error'})
        } 