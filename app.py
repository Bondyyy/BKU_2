from flask import Flask, request, jsonify, render_template, send_file
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__, template_folder='.')  # Chỉ định thư mục hiện tại là template

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['def_front', 'ok_front']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream)
    image_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        predicted_class = class_names[predicted.item()]
        probabilities = torch.softmax(outputs, dim=1)[0]
        prob_def = probabilities[0].item() * 100
        prob_ok = probabilities[1].item() * 100

    return jsonify({
        'predicted_class': predicted_class,
        'prob_def': round(prob_def, 2),
        'prob_ok': round(prob_ok, 2)
    })

@app.route('/')
def home():
    return render_template('index.html')  # Flask sẽ lấy index.html ở cùng thư mục

if __name__ == '__main__':
    app.run(debug=True)
