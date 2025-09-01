import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
def get_model():
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2)
    )
    return model.to(device)

model_path = os.path.join(os.path.dirname(__file__), 'final_model.pth')
model = get_model()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = ['def_front', 'ok_front']

# Streamlit UI
st.title("AI Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Dự đoán
    image_tensor = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        predicted_class = class_names[predicted.item()]
        probabilities = torch.softmax(outputs, dim=1)[0]
        prob_def = probabilities[0].item() * 100
        prob_ok = probabilities[1].item() * 100

    st.write(f"**Predicted class:** {predicted_class}")
    st.write(f"Probability - def_front: {prob_def:.2f}%")
    st.write(f"Probability - ok_front: {prob_ok:.2f}%")
