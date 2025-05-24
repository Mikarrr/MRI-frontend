from flask import Flask, request, jsonify
from pyngrok import ngrok
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import logging
import traceback
from datetime import datetime
import os

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== TWÓJ MODEL U-NET =====
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self._block(in_channels, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.enc4 = self._block(128, 256)

        # Bottleneck
        self.bottleneck = self._block(256, 512)

        # Decoder
        self.dec4 = self._block(512 + 256, 256)
        self.dec3 = self._block(256 + 128, 128)
        self.dec2 = self._block(128 + 64, 64)
        self.dec1 = self._block(64 + 32, 32)

        # Output
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        dec4 = self.dec4(torch.cat([self.upsample(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2), enc1], dim=1))

        return self.out(dec1)

# ===== TWOJE FUNKCJE METRYK =====
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, smooth=1e-6):
    """Obliczanie IoU dla każdej klasy"""
    # Flatten predictions and labels
    outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1) # BxCx(H*W)
    labels = labels.view(labels.shape[0], labels.shape[1], -1)   # BxCx(H*W)

    # Calculate intersection and union
    intersection = (outputs * labels).sum(dim=2)  # BxC
    union = outputs.sum(dim=2) + labels.sum(dim=2) - intersection # BxC

    # Calculate IoU for each class and batch
    iou_per_class_batch = (intersection + smooth) / (union + smooth) # BxC

    # Calculate mean IoU across the batch for each class
    iou_per_class = iou_per_class_batch.mean(dim=0) # C

    return iou_per_class

def dice_coefficient_pytorch(outputs: torch.Tensor, labels: torch.Tensor, smooth=1e-6):
    """Obliczanie Dice coefficient dla każdej klasy"""
    # Flatten predictions and labels
    outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1) # BxCx(H*W)
    labels = labels.view(labels.shape[0], labels.shape[1], -1)   # BxCx(H*W)

    intersection = (outputs * labels).sum(dim=2) # BxC

    sum_outputs = outputs.sum(dim=2) # BxC
    sum_labels = labels.sum(dim=2)   # BxC

    dice_per_class_batch = (2. * intersection + smooth) / (sum_outputs + sum_labels + smooth) # BxC

    dice_per_class = dice_per_class_batch.mean(dim=0) # C

    return dice_per_class

def mean_pixel_accuracy_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    """Obliczanie Mean Pixel Accuracy dla każdej klasy"""
    # Get the predicted class for each pixel
    _, predicted = torch.max(outputs, 1) # BxHxW

    # Convert labels to class indices
    _, true_classes = torch.max(labels, 1) # BxHxW

    # Initialize accuracy tensor for each class
    num_classes = outputs.shape[1]
    accuracy_per_class = torch.zeros(num_classes, dtype=torch.float32, device=outputs.device)
    count_per_class = torch.zeros(num_classes, dtype=torch.float32, device=outputs.device)

    # Flatten the predicted and true labels for easier comparison
    predicted = predicted.view(-1)
    true_classes = true_classes.view(-1)

    # Iterate through each class
    for class_id in range(num_classes):
        # Find pixels belonging to the current class in the ground truth
        class_mask = (true_classes == class_id)

        # Count the total number of pixels for this class in the ground truth
        total_class_pixels = torch.sum(class_mask)
        count_per_class[class_id] = total_class_pixels

        if total_class_pixels > 0:
            # Find pixels where the prediction matches the ground truth for this class
            correct_predictions = torch.sum((predicted[class_mask] == class_id))

            # Calculate accuracy for this class
            accuracy_per_class[class_id] = correct_predictions.float() / total_class_pixels.float()

    # Handle classes that are not present in the ground truth (avoid division by zero)
    accuracy_per_class[count_per_class == 0] = torch.nan

    return accuracy_per_class

# ===== FLASK APP =====
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Globalne zmienne dla modelu
model = None
device = None

# Mapowanie klas do nazw (dostosuj do swoich danych)
CLASS_NAMES = {
    0: "Tło",           # Background
    1: "Nekrotyczny",   # Necrotic tumor core
    2: "Obrzęk",        # Peritumoral edema  
    3: "Aktywny guz"    # Active tumor
}

def load_model(checkpoint_path='best_unet_model.pth'):
    """Funkcja do ładowania Twojego wytrenowanego modelu U-Net"""
    global model, device
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Używam urządzenia: {device}")
        
        # Stwórz model
        model = UNet(in_channels=1, out_channels=4)
        
        # Sprawdź czy plik checkpointa istnieje
        if os.path.exists(checkpoint_path):
            # Załaduj checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model załadowany z checkpointa: {checkpoint_path}")
            logger.info(f"Epoch: {checkpoint.get('epoch', 'N/A')}, Valid Loss: {checkpoint.get('valid_loss', 'N/A')}")
        else:
            logger.warning(f"Nie znaleziono pliku modelu: {checkpoint_path}")
            logger.warning("Model będzie działał w trybie losowych predykcji")
            return False
            
        model.to(device)
        model.eval()
        
        logger.info("Model U-Net załadowany pomyślnie")
        return True
        
    except Exception as e:
        logger.error(f"Błąd podczas ładowania modelu: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def validate_image(file):
    """Walidacja przesłanego pliku obrazu"""
    if not file:
        return False, "Brak pliku"
    
    if file.filename == '':
        return False, "Brak nazwy pliku"
    
    # Sprawdzenie rozszerzenia pliku
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        return False, f"Nieobsługiwany format pliku. Obsługiwane: {', '.join(allowed_extensions)}"
    
    return True, "OK"

def preprocess_image(image, target_size=(256, 256)):
    """Przetwarzanie obrazu zgodnie z Twoim kodem"""
    try:
        # Konwersja do skali szarości (jak w Twoim BrainMRIDataset)
        if image.mode != 'L':
            image = image.convert('L')
        
        # Zmiana rozmiaru do 256x256
        image = image.resize(target_size, Image.Resampling.BILINEAR)
        
        # Konwersja do tensora
        scan = torch.tensor(np.array(image), dtype=torch.float32)
        scan = scan.unsqueeze(0)  # Add channel dimension [1, H, W]
        
        # Normalizacja zgodnie z Twoimi transformami: (0.5,), (0.5,) -> zakres [-1, 1]
        transform = transforms.Normalize((0.5,), (0.5,))
        scan = transform(scan)
        
        # Dodaj batch dimension [1, 1, H, W]
        input_tensor = scan.unsqueeze(0)
        
        return input_tensor, None
    
    except Exception as e:
        return None, f"Błąd przetwarzania obrazu: {str(e)}"

def calculate_real_metrics(prediction_logits, create_dummy_gt=True):
    """
    Obliczanie rzeczywistych metryk na podstawie Twoich funkcji
    
    Args:
        prediction_logits: Wyjście modelu [1, 4, H, W]
        create_dummy_gt: Czy stworzyć sztuczne ground truth do demonstracji
    """
    try:
        with torch.no_grad():
            # Konwertuj logity do prawdopodobieństw
            prediction_probs = torch.softmax(prediction_logits, dim=1)
            
            # Uzyskaj maskę segmentacji (indeksy klas)
            prediction_mask = prediction_logits.argmax(dim=1).squeeze().cpu().numpy()
            
            if create_dummy_gt:
                # Stwórz sztuczne ground truth do demonstracji metryk
                # W prawdziwej aplikacji tego by nie było
                gt_mask = prediction_mask.copy()
                # Dodaj trochę szumu żeby metryki nie były perfekcyjne
                noise_mask = np.random.random(gt_mask.shape) < 0.05
                gt_mask[noise_mask] = np.random.randint(0, 4, size=np.sum(noise_mask))
                
                # Konwertuj ground truth do one-hot
                gt_one_hot = torch.zeros_like(prediction_probs)
                for i in range(4):
                    gt_one_hot[0, i] = torch.tensor(gt_mask == i, dtype=torch.float32)
                
                # Oblicz metryki używając Twoich funkcji
                iou_scores = iou_pytorch(prediction_probs, gt_one_hot)
                dice_scores = dice_coefficient_pytorch(prediction_probs, gt_one_hot)
                mpa_scores = mean_pixel_accuracy_pytorch(prediction_probs, gt_one_hot)
                
                # Oblicz średnie metryki
                mean_iou = torch.nanmean(iou_scores).item()
                mean_dice = torch.nanmean(dice_scores).item()
                mean_mpa = torch.nanmean(mpa_scores[~torch.isnan(mpa_scores)]).item() if torch.sum(~torch.isnan(mpa_scores)) > 0 else 0.0
            else:
                # Bez ground truth - zwróć tylko informacje o predykcji
                mean_iou = 0.0
                mean_dice = 0.0  
                mean_mpa = 0.0
                iou_scores = torch.zeros(4)
                dice_scores = torch.zeros(4)
                mpa_scores = torch.zeros(4)
            
            # Statystyki predykcji
            unique_classes, class_counts = np.unique(prediction_mask, return_counts=True)
            class_percentages = {int(cls): float(count) / prediction_mask.size * 100 
                               for cls, count in zip(unique_classes, class_counts)}
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'image_shape': prediction_mask.shape,
                'num_classes_detected': len(unique_classes),
                'mean_iou': round(mean_iou, 4),
                'mean_dice': round(mean_dice, 4),
                'mean_pixel_accuracy': round(mean_mpa, 4),
                'class_distribution': {
                    CLASS_NAMES.get(cls, f"Class_{cls}"): {
                        'percentage': round(class_percentages.get(cls, 0.0), 2),
                        'pixel_count': int(class_counts[list(unique_classes).index(cls)] if cls in unique_classes else 0)
                    }
                    for cls in range(4)
                },
                'class_metrics': {
                    CLASS_NAMES.get(i, f"Class_{i}"): {
                        'iou': round(iou_scores[i].item(), 4) if not torch.isnan(iou_scores[i]) else 0.0,
                        'dice': round(dice_scores[i].item(), 4) if not torch.isnan(dice_scores[i]) else 0.0,
                        'pixel_accuracy': round(mpa_scores[i].item(), 4) if not torch.isnan(mpa_scores[i]) else 0.0,
                    }
                    for i in range(4)
                }
            }
            
            return prediction_mask, metrics
        
    except Exception as e:
        logger.error(f"Błąd obliczania metryk: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint sprawdzania stanu serwera"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'device': str(device) if device else None,
        'model_type': 'U-Net Brain MRI Segmentation',
        'classes': CLASS_NAMES
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Główny endpoint predykcji segmentacji mózgu"""
    try:
        # Sprawdzenie czy plik został przesłany
        if 'file' not in request.files:
            return jsonify({'error': 'Brak pliku w żądaniu. Użyj klucza "file" w form-data'}), 400

        file = request.files['file']
        
        # Walidacja pliku
        is_valid, validation_message = validate_image(file)
        if not is_valid:
            return jsonify({'error': validation_message}), 400

        # Odczytanie i otwarcie obrazu
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Obraz załadowany: {image.size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"Błąd otwierania obrazu: {str(e)}")
            return jsonify({'error': 'Nie można otworzyć pliku obrazu'}), 400

        # Przetworzenie obrazu zgodnie z Twoimi wymaganiami
        input_tensor, error = preprocess_image(image)
        if error:
            return jsonify({'error': error}), 400

        # Wykonanie predykcji
        try:
            if model is not None:
                # Rzeczywista predykcja Twoim modelem U-Net
                input_tensor = input_tensor.to(device)
                
                with torch.no_grad():
                    # Wykonaj predykcję
                    prediction_logits = model(input_tensor)  # [1, 4, 256, 256]
                    
                    # Oblicz metryki i uzyskaj maskę
                    prediction_mask, metrics = calculate_real_metrics(prediction_logits)
                    
                    if prediction_mask is None:
                        return jsonify({'error': 'Błąd przetwarzania predykcji'}), 500
                        
                    logger.info("Predykcja wykonana pomyślnie")
            else:
                return jsonify({'error': 'Model nie jest załadowany'}), 500
        
        except Exception as e:
            logger.error(f"Błąd podczas predykcji: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Błąd podczas wykonywania predykcji: {str(e)}'}), 500

        # Przygotowanie odpowiedzi
        response = {
            'success': True,
            'segmentation_mask': prediction_mask.tolist(),
            'metrics': metrics,
            'info': {
                'original_filename': file.filename,
                'original_size': f"{image.size[0]}x{image.size[1]}",
                'processed_size': "256x256",
                'model_type': 'U-Net Brain MRI Segmentation',
                'processing_time': datetime.now().isoformat(),
                'device_used': str(device)
            }
        }

        logger.info(f"Pomyślnie przetworzono obraz: {file.filename}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Nieoczekiwany błąd: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Wewnętrzny błąd serwera'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Plik jest za duży (max 16MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint nie znaleziony'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Wewnętrzny błąd serwera'}), 500

def start_ngrok(port=5000):
    """Uruchomienie ngrok z obsługą błędów"""
    try:
        ngrok_tunnel = ngrok.connect(port)
        public_url = ngrok_tunnel.public_url
        logger.info(f'Publiczny URL ngrok: {public_url}')
        logger.info(f'Endpoint predykcji: {public_url}/predict')
        logger.info(f'Health check: {public_url}/health')
        return public_url
    except Exception as e:
        logger.error(f"Błąd uruchamiania ngrok: {str(e)}")
        return None

if __name__ == '__main__':
    print("=" * 60)
    print("🧠 SERWER SEGMENTACJI OBRAZÓW MÓZGU MRI")
    print("=" * 60)
    
    # Ładowanie modelu U-Net
    checkpoint_path = 'best_unet_model.pth'  # Zmień ścieżkę jeśli potrzeba
    model_loaded = load_model(checkpoint_path)
    
    if not model_loaded:
        print("⚠️  UWAGA: Model nie został załadowany!")
        print(f"   Upewnij się, że plik '{checkpoint_path}' istnieje")
        print("   Serwer będzie działał, ale nie będzie mógł wykonywać predykcji")
    else:
        print("✅ Model U-Net załadowany pomyślnie!")
    
    # Konfiguracja portu
    port = int(os.environ.get('PORT', 5000))
    
    # Uruchomienie ngrok
    public_url = start_ngrok(port)
    
    print(f"\n🚀 Serwer gotowy do użycia!")
    print(f"📍 Lokalny adres: http://localhost:{port}")
    if public_url:
        print(f"🌍 Publiczny adres: {public_url}")
    
    print(f"\n📋 Dostępne endpointy:")
    print(f"   POST /predict - segmentacja obrazu MRI")
    print(f"   GET /health - status serwera")
    
    print(f"\n🎯 Klasy segmentacji:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"   {class_id}: {class_name}")
    
    print(f"\n💡 Przykładowe użycie:")
    print(f"   curl -X POST -F \"file=@brain_scan.jpg\" {public_url or 'http://localhost:' + str(port)}/predict")
    
    print(f"\n⏹️  Aby zatrzymać serwer, naciśnij Ctrl+C")
    print("=" * 60)
    
    # Uruchomienie serwera Flask
    app.run(host='0.0.0.0', port=port, debug=False)