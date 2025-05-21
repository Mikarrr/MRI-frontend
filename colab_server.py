from flask import Flask, request, jsonify
from pyngrok import ngrok
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import torch.nn as nn
import torch.nn.functional as F

# Definicja modelu U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        
        # Middle
        self.middle = DoubleConv(512, 1024)
        
        # Decoder
        self.decoder4 = DoubleConv(1024 + 512, 512)
        self.decoder3 = DoubleConv(512 + 256, 256)
        self.decoder2 = DoubleConv(256 + 128, 128)
        self.decoder1 = DoubleConv(128 + 64, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Upsampling
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool(enc2)
        
        enc3 = self.encoder3(x)
        x = self.pool(enc3)
        
        enc4 = self.encoder4(x)
        x = self.pool(enc4)
        
        # Middle
        x = self.middle(x)
        
        # Decoder
        x = self.up4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.decoder4(x)
        
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)
        
        # Final layer
        x = self.final(x)
        
        return x

# Funkcja do obliczania metryk
def calculate_metrics(pred, target):
    # Konwersja do np.array jeśli są tensorami
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
        
    # Upewnienie się, że mają odpowiedni kształt
    if len(pred.shape) > 2:
        pred = np.argmax(pred, axis=1)[0]  # Dla batch size 1
    if len(target.shape) > 2:
        target = target[0]
    
    # Liczba klas (włącznie z tłem)
    num_classes = 4
    
    # Metryki ogólne
    accuracy = np.mean(pred == target)
    
    # Inicjalizacja metryk dla poszczególnych klas
    class_metrics = {}
    total_dice = 0
    total_iou = 0
    classes_present = 0
    
    # Obliczanie metryk dla każdej klasy
    for c in range(num_classes):
        # Maski binarne dla klasy c
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        
        # Warunek sprawdzający czy klasa występuje w target
        if np.sum(target_c) == 0:
            continue
        
        # Obliczanie metryk
        intersection = np.sum(pred_c * target_c)
        pred_area = np.sum(pred_c)
        target_area = np.sum(target_c)
        
        # Dice coefficient
        dice = (2.0 * intersection) / (pred_area + target_area + 1e-8)
        
        # IoU (Jaccard index)
        union = pred_area + target_area - intersection
        iou = intersection / (union + 1e-8)
        
        # Precision i Recall
        precision = intersection / (pred_area + 1e-8)
        recall = intersection / (target_area + 1e-8)
        
        # Dodanie do ogólnych metryk
        total_dice += dice
        total_iou += iou
        classes_present += 1
        
        # Zapisanie metryk dla klasy
        class_metrics[str(c)] = {
            'dice': float(dice),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall)
        }
    
    # Średnie metryki po wszystkich klasach
    mean_dice = total_dice / max(1, classes_present)
    mean_iou = total_iou / max(1, classes_present)
    
    metrics = {
        'dice': float(mean_dice),
        'iou': float(mean_iou),
        'accuracy': float(accuracy),
        'class_metrics': class_metrics
    }
    
    return metrics

# Inicjalizacja aplikacji Flask
app = Flask(__name__)

# Załadowanie modelu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1, out_channels=4).to(device)

# Ładowanie wag modelu - odkomentuj to gdy będziesz mieć plik z wagami
# model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    try:
        file = request.files['file']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Przetwarzanie obrazu
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        input_tensor = transform(image.convert('L')).unsqueeze(0).to(device)
        
        # Wykonanie predykcji
        with torch.no_grad():
            prediction = model(input_tensor)
            mask = prediction.argmax(1).squeeze().cpu().numpy()
        
        # Obliczanie metryk
        # W rzeczywistości potrzebowalibyśmy tu ground truth do metryk
        # Dla demonstracji generujemy losowe ground truth
        dummy_target = np.random.randint(0, 4, size=mask.shape)
        metrics = calculate_metrics(mask, dummy_target)
        
        return jsonify({
            'mask': mask.tolist(),
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Uruchomienie ngrok do udostępnienia aplikacji
public_url = ngrok.connect(5000).public_url
print(f'Publiczny URL serwera: {public_url}')
print(f'Endpoint do predykcji: {public_url}/predict')

# Uruchomienie serwera Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)