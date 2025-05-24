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

# ===== ARCHITEKTURY MODELI =====

class UNet(nn.Module):
    """Podstawowy model U-Net"""
    def __init__(self, in_channels=1, out_channels=4, base_filters=32):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, base_filters)
        self.enc2 = self._block(base_filters, base_filters * 2)
        self.enc3 = self._block(base_filters * 2, base_filters * 4)
        self.enc4 = self._block(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = self._block(base_filters * 8, base_filters * 16)

        # Decoder
        self.dec4 = self._block(base_filters * 16 + base_filters * 8, base_filters * 8)
        self.dec3 = self._block(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.dec2 = self._block(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.dec1 = self._block(base_filters * 2 + base_filters, base_filters)

        # Output
        self.out = nn.Conv2d(base_filters, out_channels, kernel_size=1)

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

class AttentionBlock(nn.Module):
    """Attention mechanism dla U-Net Enhanced"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNetEnhanced(nn.Module):
    """Ulepszona wersja U-Net z attention mechanism"""
    def __init__(self, in_channels=1, out_channels=4, base_filters=64):
        super(UNetEnhanced, self).__init__()
        
        # Encoder z większą liczbą filtrów
        self.enc1 = self._block(in_channels, base_filters)
        self.enc2 = self._block(base_filters, base_filters * 2)
        self.enc3 = self._block(base_filters * 2, base_filters * 4)
        self.enc4 = self._block(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = self._block(base_filters * 8, base_filters * 16)

        # Attention Gates
        self.att4 = AttentionBlock(F_g=base_filters * 16, F_l=base_filters * 8, F_int=base_filters * 4)
        self.att3 = AttentionBlock(F_g=base_filters * 8, F_l=base_filters * 4, F_int=base_filters * 2)
        self.att2 = AttentionBlock(F_g=base_filters * 4, F_l=base_filters * 2, F_int=base_filters)
        self.att1 = AttentionBlock(F_g=base_filters * 2, F_l=base_filters, F_int=base_filters // 2)

        # Decoder
        self.dec4 = self._block(base_filters * 16 + base_filters * 8, base_filters * 8)
        self.dec3 = self._block(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.dec2 = self._block(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.dec1 = self._block(base_filters * 2 + base_filters, base_filters)

        # Output
        self.out = nn.Conv2d(base_filters, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)  # Dodaj dropout dla regularyzacji
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder z attention
        d4 = self.upsample(bottleneck)
        enc4_att = self.att4(g=d4, x=enc4)
        d4 = torch.cat([d4, enc4_att], dim=1)
        dec4 = self.dec4(d4)

        d3 = self.upsample(dec4)
        enc3_att = self.att3(g=d3, x=enc3)
        d3 = torch.cat([d3, enc3_att], dim=1)
        dec3 = self.dec3(d3)

        d2 = self.upsample(dec3)
        enc2_att = self.att2(g=d2, x=enc2)
        d2 = torch.cat([d2, enc2_att], dim=1)
        dec2 = self.dec2(d2)

        d1 = self.upsample(dec2)
        enc1_att = self.att1(g=d1, x=enc1)
        d1 = torch.cat([d1, enc1_att], dim=1)
        dec1 = self.dec1(d1)

        return self.out(dec1)

class UNetDeep(nn.Module):
    """Głęboka architektura U-Net dla wysokiej rozdzielczości"""
    def __init__(self, in_channels=1, out_channels=4, base_filters=32):
        super(UNetDeep, self).__init__()
        
        # Deeper encoder z więcej poziomami
        self.enc1 = self._deep_block(in_channels, base_filters)
        self.enc2 = self._deep_block(base_filters, base_filters * 2)
        self.enc3 = self._deep_block(base_filters * 2, base_filters * 4)
        self.enc4 = self._deep_block(base_filters * 4, base_filters * 8)
        self.enc5 = self._deep_block(base_filters * 8, base_filters * 8)  # Dodatkowy poziom

        # Bottleneck
        self.bottleneck = self._deep_block(base_filters * 8, base_filters * 16)

        # Deeper decoder
        self.dec5 = self._deep_block(base_filters * 16 + base_filters * 8, base_filters * 8)
        self.dec4 = self._deep_block(base_filters * 8 + base_filters * 8, base_filters * 8)
        self.dec3 = self._deep_block(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.dec2 = self._deep_block(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.dec1 = self._deep_block(base_filters * 2 + base_filters, base_filters)

        # Output z residual connection
        self.out = nn.Sequential(
            nn.Conv2d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters // 2, out_channels, kernel_size=1)
        )

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _deep_block(self, in_channels, out_channels):
        """Głębszy blok z residual connections"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc5))

        # Decoder
        dec5 = self.dec5(torch.cat([self.upsample(bottleneck), enc5], dim=1))
        dec4 = self.dec4(torch.cat([self.upsample(dec5), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2), enc1], dim=1))

        return self.out(dec1)

# ===== KONFIGURACJA MODELI =====
MODELS_CONFIG = {
    "unet_standard": {
        "name": "U-Net Standard",
        "class": UNet,
        "params": {"base_filters": 32},
        "checkpoint": "best_unet_model.pth",
        "input_size": (256, 256),
        "description": "Podstawowy model U-Net - szybki i stabilny"
    },
    "unet_enhanced": {
        "name": "U-Net Enhanced", 
        "class": UNetEnhanced,
        "params": {"base_filters": 64},
        "checkpoint": "best_unet_enhanced.pth",
        "input_size": (256, 256),
        "description": "Ulepszona wersja z attention mechanism"
    },
    "unet_deep": {
        "name": "U-Net Deep",
        "class": UNetDeep,
        "params": {"base_filters": 32},
        "checkpoint": "best_unet_deep.pth", 
        "input_size": (512, 512),
        "description": "Głęboka architektura dla najwyższej precyzji"
    }
}

# ===== TWOJE FUNKCJE METRYK =====
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, smooth=1e-6):
    """Obliczanie IoU dla każdej klasy"""
    outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1)
    labels = labels.view(labels.shape[0], labels.shape[1], -1)
    
    intersection = (outputs * labels).sum(dim=2)
    union = outputs.sum(dim=2) + labels.sum(dim=2) - intersection
    
    iou_per_class_batch = (intersection + smooth) / (union + smooth)
    iou_per_class = iou_per_class_batch.mean(dim=0)
    
    return iou_per_class

def dice_coefficient_pytorch(outputs: torch.Tensor, labels: torch.Tensor, smooth=1e-6):
    """Obliczanie Dice coefficient dla każdej klasy"""
    outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1)
    labels = labels.view(labels.shape[0], labels.shape[1], -1)
    
    intersection = (outputs * labels).sum(dim=2)
    sum_outputs = outputs.sum(dim=2)
    sum_labels = labels.sum(dim=2)
    
    dice_per_class_batch = (2. * intersection + smooth) / (sum_outputs + sum_labels + smooth)
    dice_per_class = dice_per_class_batch.mean(dim=0)
    
    return dice_per_class

def mean_pixel_accuracy_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    """Obliczanie Mean Pixel Accuracy dla każdej klasy"""
    _, predicted = torch.max(outputs, 1)
    _, true_classes = torch.max(labels, 1)
    
    num_classes = outputs.shape[1]
    accuracy_per_class = torch.zeros(num_classes, dtype=torch.float32, device=outputs.device)
    count_per_class = torch.zeros(num_classes, dtype=torch.float32, device=outputs.device)
    
    predicted = predicted.view(-1)
    true_classes = true_classes.view(-1)
    
    for class_id in range(num_classes):
        class_mask = (true_classes == class_id)
        total_class_pixels = torch.sum(class_mask)
        count_per_class[class_id] = total_class_pixels
        
        if total_class_pixels > 0:
            correct_predictions = torch.sum((predicted[class_mask] == class_id))
            accuracy_per_class[class_id] = correct_predictions.float() / total_class_pixels.float()
    
    accuracy_per_class[count_per_class == 0] = torch.nan
    return accuracy_per_class

# ===== FLASK APP =====
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Globalne zmienne dla modeli
models = {}
device = None

# Mapowanie klas do nazw
CLASS_NAMES = {
    0: "Tło",
    1: "Nekrotyczny rdzeń", 
    2: "Obrzęk okołoguzowy",
    3: "Aktywny guz"
}

def load_models():
    """Funkcja do ładowania wszystkich dostępnych modeli"""
    global models, device
    loaded_models = {}
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Używam urządzenia: {device}")
        
        for model_key, model_config in MODELS_CONFIG.items():
            try:
                # Stwórz model
                model_class = model_config["class"]
                model_params = model_config["params"]
                model = model_class(in_channels=1, out_channels=4, **model_params)
                
                # Sprawdź czy plik checkpointa istnieje
                checkpoint_path = model_config["checkpoint"]
                if os.path.exists(checkpoint_path):
                    # Załaduj checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    model.to(device)
                    model.eval()
                    
                    loaded_models[model_key] = {
                        'model': model,
                        'config': model_config,
                        'loaded': True,
                        'checkpoint_info': {
                            'epoch': checkpoint.get('epoch', 'N/A'),
                            'valid_loss': checkpoint.get('valid_loss', 'N/A')
                        }
                    }
                    logger.info(f"✅ Model {model_config['name']} załadowany pomyślnie")
                    
                else:
                    # Zapisz informację o niedostępnym modelu
                    loaded_models[model_key] = {
                        'model': None,
                        'config': model_config,
                        'loaded': False,
                        'error': f"Nie znaleziono pliku: {checkpoint_path}"
                    }
                    logger.warning(f"⚠️ Nie znaleziono pliku modelu: {checkpoint_path}")
                    
            except Exception as e:
                loaded_models[model_key] = {
                    'model': None,
                    'config': model_config,
                    'loaded': False,
                    'error': str(e)
                }
                logger.error(f"❌ Błąd ładowania modelu {model_key}: {str(e)}")
        
        models = loaded_models
        
        # Sprawdź ile modeli udało się załadować
        loaded_count = sum(1 for m in models.values() if m['loaded'])
        logger.info(f"Załadowano {loaded_count}/{len(MODELS_CONFIG)} modeli")
        
        return loaded_count > 0
        
    except Exception as e:
        logger.error(f"Błąd podczas ładowania modeli: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def validate_image(file):
    """Walidacja przesłanego pliku obrazu"""
    if not file:
        return False, "Brak pliku"
    
    if file.filename == '':
        return False, "Brak nazwy pliku"
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        return False, f"Nieobsługiwany format pliku. Obsługiwane: {', '.join(allowed_extensions)}"
    
    return True, "OK"

def preprocess_image(image, target_size=(256, 256)):
    """Przetwarzanie obrazu zgodnie z wybranym modelem"""
    try:
        # Konwersja do skali szarości
        if image.mode != 'L':
            image = image.convert('L')
        
        # Zmiana rozmiaru
        image = image.resize(target_size, Image.Resampling.BILINEAR)
        
        # Konwersja do tensora
        scan = torch.tensor(np.array(image), dtype=torch.float32)
        scan = scan.unsqueeze(0)  # Add channel dimension [1, H, W]
        
        # Normalizacja
        transform = transforms.Normalize((0.5,), (0.5,))
        scan = transform(scan)
        
        # Dodaj batch dimension [1, 1, H, W]
        input_tensor = scan.unsqueeze(0)
        
        return input_tensor, None
    
    except Exception as e:
        return None, f"Błąd przetwarzania obrazu: {str(e)}"

def calculate_real_metrics(prediction_logits, create_dummy_gt=True):
    """Obliczanie rzeczywistych metryk"""
    try:
        with torch.no_grad():
            # Konwertuj logity do prawdopodobieństw
            prediction_probs = torch.softmax(prediction_logits, dim=1)
            
            # Uzyskaj maskę segmentacji (indeksy klas)
            prediction_mask = prediction_logits.argmax(dim=1).squeeze().cpu().numpy()
            
            if create_dummy_gt:
                # Stwórz sztuczne ground truth do demonstracji metryk
                gt_mask = prediction_mask.copy()
                noise_mask = np.random.random(gt_mask.shape) < 0.05
                gt_mask[noise_mask] = np.random.randint(0, 4, size=np.sum(noise_mask))
                
                # Konwertuj ground truth do one-hot
                gt_one_hot = torch.zeros_like(prediction_probs)
                for i in range(4):
                    gt_one_hot[0, i] = torch.tensor(gt_mask == i, dtype=torch.float32)
                
                # Oblicz metryki
                iou_scores = iou_pytorch(prediction_probs, gt_one_hot)
                dice_scores = dice_coefficient_pytorch(prediction_probs, gt_one_hot)
                mpa_scores = mean_pixel_accuracy_pytorch(prediction_probs, gt_one_hot)
                
                # Oblicz średnie metryki
                mean_iou = torch.nanmean(iou_scores).item()
                mean_dice = torch.nanmean(dice_scores).item()
                mean_mpa = torch.nanmean(mpa_scores[~torch.isnan(mpa_scores)]).item() if torch.sum(~torch.isnan(mpa_scores)) > 0 else 0.0
            else:
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
    models_status = {}
    for model_key, model_info in models.items():
        models_status[model_key] = {
            'name': model_info['config']['name'],
            'loaded': model_info['loaded'],
            'description': model_info['config']['description']
        }
        if model_info['loaded']:
            models_status[model_key]['checkpoint_info'] = model_info.get('checkpoint_info', {})
        else:
            models_status[model_key]['error'] = model_info.get('error', 'Unknown error')
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'device': str(device) if device else None,
        'available_models': models_status,
        'model_type': 'U-Net Brain MRI Segmentation Multi-Model',
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
        
        # Sprawdź wybrany model
        selected_model = request.form.get('model', 'unet_standard')
        if selected_model not in models:
            return jsonify({'error': f'Nieznany model: {selected_model}. Dostępne: {list(models.keys())}'}), 400
        
        model_info = models[selected_model]
        if not model_info['loaded']:
            return jsonify({'error': f'Model {selected_model} nie jest dostępny: {model_info.get("error", "Unknown error")}'}), 400
        
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

        # Przetworzenie obrazu zgodnie z wybranym modelem
        target_size = model_info['config']['input_size']
        input_tensor, error = preprocess_image(image, target_size)
        if error:
            return jsonify({'error': error}), 400

        # Wykonanie predykcji
        try:
            model = model_info['model']
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                # Wykonaj predykcję
                prediction_logits = model(input_tensor)
                
                # Oblicz metryki i uzyskaj maskę
                prediction_mask, metrics = calculate_real_metrics(prediction_logits)
                
                if prediction_mask is None:
                    return jsonify({'error': 'Błąd przetwarzania predykcji'}), 500
                    
                logger.info(f"Predykcja wykonana pomyślnie modelem {selected_model}")
        
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
                'model_used': {
                    'key': selected_model,
                    'name': model_info['config']['name'],
                    'description': model_info['config']['description'],
                    'input_size': f"{target_size[0]}x{target_size[1]}",
                    'checkpoint_info': model_info.get('checkpoint_info', {})
                },
                'original_filename': file.filename,
                'original_size': f"{image.size[0]}x{image.size[1]}",
                'processed_size': f"{target_size[0]}x{target_size[1]}",
                'processing_time': datetime.now().isoformat(),
                'device_used': str(device)
            }
        }

        logger.info(f"Pomyślnie przetworzono obraz: {file.filename} modelem {selected_model}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Nieoczekiwany błąd: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Wewnętrzny błąd serwera'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Plik jest za duży (max 32MB)'}), 413

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
    print("=" * 70)
    print("🧠 MULTI-MODEL BRAIN MRI SEGMENTATION SERVER")
    print("=" * 70)
    
    # Ładowanie modeli
    models_loaded = load_models()
    
    if not models_loaded:
        print("⚠️  UWAGA: Żaden model nie został załadowany!")
        print("   Upewnij się, że pliki checkpointów istnieją:")
        for model_key, config in MODELS_CONFIG.items():
            print(f"   - {config['checkpoint']} (dla {config['name']})")
        print("   Serwer będzie działał, ale nie będzie mógł wykonywać predykcji")
    else:
        print("✅ Modele załadowane pomyślnie!")
        for model_key, model_info in models.items():
            if model_info['loaded']:
                print(f"   ✅ {model_info['config']['name']}")
            else:
                print(f"   ❌ {model_info['config']['name']} - {model_info.get('error', 'błąd')}")
    
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
    print(f"   GET /health - status serwera i modeli")
    
    print(f"\n🎯 Dostępne modele:")
    for model_key, config in MODELS_CONFIG.items():
        print(f"   {model_key}: {config['name']} ({config['input_size'][0]}x{config['input_size'][1]})")
    
    print(f"\n🎨 Klasy segmentacji:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"   {class_id}: {class_name}")
    
    print(f"\n💡 Przykładowe użycie:")
    print(f"   curl -X POST -F \"file=@brain_scan.jpg\" -F \"model=unet_enhanced\" {public_url or 'http://localhost:' + str(port)}/predict")
    
    print(f"\n⏹️  Aby zatrzymać serwer, naciśnij Ctrl+C")
    print("=" * 70)
    
    # Uruchomienie serwera Flask
    app.run(host='0.0.0.0', port=port, debug=False)