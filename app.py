import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- Thiết bị ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hàm tạo model ---
def get_model():
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2)
    )
    return model.to(device)

# --- Cache model để chỉ load 1 lần ---
@st.cache_resource
def load_model():
    model = get_model()
    model.load_state_dict(torch.load('final_model.pth', map_location=device))
    model.eval()
    return model

model = load_model()

# --- Transform ảnh ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Class names ---
class_names = ['def_front', 'ok_front']

# --- Streamlit UI ---
st.title("AI Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")  # Đảm bảo 3 channel
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # --- Dự đoán ---
    image_tensor = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        predicted_class = class_names[predicted.item()]
        probabilities = torch.softmax(outputs, dim=1)[0]
        prob_def = probabilities[0].item() * 100
        prob_ok = probabilities[1].item() * 100

    # --- Hiển thị kết quả ---
    st.write(f"**Predicted class:** {predicted_class}")
    st.write(f"Probability - def_front: {prob_def:.2f}%")
    st.write(f"Probability - ok_front: {prob_ok:.2f}%")

