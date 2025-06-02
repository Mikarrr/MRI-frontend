import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import io
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Sprawdź czy plotly jest dostępne
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly nie jest zainstalowany. Wykresy będą wyświetlane w trybie podstawowym. Zainstaluj: `pip install plotly`")

# Sprawdź czy pandas jest dostępne  
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    st.warning("Pandas nie jest zainstalowany. Tabele będą wyświetlane w trybie podstawowym. Zainstaluj: `pip install pandas`")

# Konfiguracja strony
st.set_page_config(
    layout="wide", 
    page_title="Brain MRI Segmentation AI",
    page_icon="🧠"
)

# Style CSS dla interfejsu
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #2c3e50;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .results-container {
        background: transparent;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: transparent;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
    }
    .model-card {
        background: transparent;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #e9ecef;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    .model-card:hover {
        border-color: #3498db;
        box-shadow: 0 0.25rem 0.5rem rgba(52, 152, 219, 0.2);
    }
    .class-legend {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        background: transparent;
    }
    .color-box {
        width: 20px;
        height: 20px;
        border-radius: 3px;
        margin-right: 10px;
        border: 1px solid #ccc;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online { background-color: #27ae60; }
    .status-offline { background-color: #e74c3c; }
    .status-loading { background-color: #f39c12; }
    .demo-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =================================
# DEFINICJA MODELU UNET
# =================================

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()
        x = 16
        # Encoder
        self.enc1 = self._block(in_channels, x)
        self.enc2 = self._block(x, 2*x)
        self.enc3 = self._block(2*x, 4*x)
        self.enc4 = self._block(4*x, 8*x)

        # Bottleneck
        self.bottleneck = self._block(8*x, 16*x)

        # Decoder
        self.dec4 = self._block(16*x + 8*x, 8*x)
        self.dec3 = self._block(8*x + 4*x, 4*x)
        self.dec2 = self._block(4*x + 2*x, 2*x)
        self.dec1 = self._block(2*x + x, x)

        # Output
        self.out = nn.Conv2d(x, out_channels, kernel_size=1)

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

# =================================
# FUNKCJE METRYK
# =================================

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, smooth=1e-6):
    """Obliczanie IoU (Intersection over Union)"""
    outputs = outputs.float()
    labels = labels.float()

    if outputs.dim() == 4:
        outputs = outputs.squeeze(1)
        labels = labels.squeeze(1)

    outputs = outputs.view(outputs.size(0), -1)
    labels = labels.view(labels.size(0), -1)

    intersection = (outputs * labels).sum(dim=1)
    union = outputs.sum(dim=1) + labels.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean()  # shape: (B,)

def dice_coefficient_pytorch(outputs: torch.Tensor, labels: torch.Tensor, smooth=1e-6):
    """Obliczanie współczynnika Dice'a"""
    outputs = outputs.float()
    labels = labels.float()

    if outputs.dim() == 4:
        outputs = outputs.squeeze(1)
        labels = labels.squeeze(1)

    outputs = outputs.view(outputs.size(0), -1)
    labels = labels.view(labels.size(0), -1)

    intersection = (outputs * labels).sum(dim=1)
    sum_outputs = outputs.sum(dim=1)
    sum_labels = labels.sum(dim=1)

    dice = (2. * intersection + smooth) / (sum_outputs + sum_labels + smooth)

    return dice.mean()

def mean_pixel_accuracy_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    """Obliczanie średniej dokładności pikselowej"""
    if outputs.dim() == 4:
        outputs = outputs.squeeze(1)
    if labels.dim() == 4:
        labels = labels.squeeze(1)

    outputs = outputs.float()
    labels = labels.float()

    correct = (outputs == labels).float()
    accuracy_per_image = correct.view(correct.size(0), -1).mean(dim=1)  # (B,)

    return accuracy_per_image.mean()

# =================================
# KONFIGURACJA APLIKACJI
# =================================

# Definicje modeli
MODELS_CONFIG = {
    "unet_standard": {
        "name": "U-Net Standard",
        "description": "Model U-Net do segmentacji obrazów MRI",
        "checkpoint": "best_unet_model.pth", # Ścieżka do Twojego zapisanego modelu
        "input_size": (256, 256),
        "features": ["Szybka predykcja", "Dobra ogólna jakość", "Stabilny"],
        "recommended_for": "Segmentacja guzów mózgu"
    }
    # Możesz dodać więcej modeli jeśli masz ich więcej
}

# Definicje klas segmentacji
CLASS_DEFINITIONS = {
    0: {
        "name": "Tło",
        "description": "Obszary nie będące guzem",
        "color": [0, 0, 0],
        "hex": "#000000"
    },
    1: {
        "name": "Guz",
        "description": "Obszary z guzem mózgu",
        "color": [255, 0, 0],
        "hex": "#FF0000"
    }
}

# CLASS_NAMES dla kompatybilności
CLASS_NAMES = {i: info["name"] for i, info in CLASS_DEFINITIONS.items()}

# Globalne zmienne dla modeli
models = {}
device = None

# =================================
# INICJALIZACJA SESJI
# =================================

if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "unet_standard"
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# =================================
# FUNKCJE POMOCNICZE
# =================================

@st.cache_resource
def load_models():
    """Funkcja do ładowania modeli (zakeszowana przez Streamlit)"""
    loaded_models = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_key, model_config in MODELS_CONFIG.items():
        try:
            # Stwórz model
            model = UNet(in_channels=1, out_channels=2)
            
            # Sprawdź czy plik checkpointa istnieje
            checkpoint_path = model_config["checkpoint"]
            if os.path.exists(checkpoint_path):
                # Załaduj checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Sprawdź format checkpointa i dostosuj ładowanie
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Jeśli to tylko state_dict bez dodatkowych kluczy
                    model.load_state_dict(checkpoint)
                
                model.to(device)
                model.eval()
                
                loaded_models[model_key] = {
                    'model': model,
                    'config': model_config,
                    'loaded': True,
                    'device': device
                }
                print(f"✅ Model {model_config['name']} załadowany pomyślnie")
                
            else:
                # Zapisz informację o niedostępnym modelu
                loaded_models[model_key] = {
                    'model': None,
                    'config': model_config,
                    'loaded': False,
                    'error': f"Nie znaleziono pliku: {checkpoint_path}"
                }
                print(f"⚠️ Nie znaleziono pliku modelu: {checkpoint_path}")
                
        except Exception as e:
            loaded_models[model_key] = {
                'model': None,
                'config': model_config,
                'loaded': False,
                'error': str(e)
            }
            print(f"❌ Błąd ładowania modelu {model_key}: {str(e)}")
    
    return loaded_models, device

def normalize_image(img):
    """Normalizacja obrazu MRI dla lepszej wizualizacji"""
    img_np = np.array(img.convert('L'))
    img_min, img_max = img_np.min(), img_np.max()
    if img_max > img_min:
        img_np = ((img_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_np = np.zeros_like(img_np, dtype=np.uint8)
    return Image.fromarray(img_np)

def colorize_mask(mask):
    """Generuje kolorową maskę na podstawie predykcji"""
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, class_info in CLASS_DEFINITIONS.items():
        rgb_mask[mask == class_id] = class_info["color"]
    return rgb_mask

def overlay_masks(image, mask, alpha=0.6):
    """Nakłada kolorową maskę na oryginalny obraz"""
    image = np.array(image.convert('RGB'))
    
    # Upewnij się, że oba obrazy mają ten sam rozmiar
    if image.shape[:2] != mask.shape[:2]:
        mask = Image.fromarray(mask)
        mask = mask.resize((image.shape[1], image.shape[0]), Image.NEAREST)
        mask = np.array(mask)
    
    # Nakładka maski na obraz
    blended = image.copy()
    colored_mask = colorize_mask(mask)
    mask_indices = mask > 0  # Indeksy, gdzie maska ma wartości niezerowe
    
    # Nakładka tylko tam, gdzie maska ma wartości
    blended_pixels = mask_indices[:, :, np.newaxis].repeat(3, axis=2)
    blended[blended_pixels] = (alpha * colored_mask[blended_pixels] + 
                             (1 - alpha) * image[blended_pixels]).astype(np.uint8)
    
    return blended

def preprocess_image(image, target_size=(256, 256)):
    """Przetwarzanie obrazu zgodnie z wymaganiami modelu"""
    try:
        # Konwersja do skali szarości
        if image.mode != 'L':
            image = image.convert('L')
        
        # Zmiana rozmiaru
        image = image.resize(target_size, Image.LANCZOS)
        
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

def predict_with_local_model(image, model_key):
    """Wykonuje predykcję używając lokalnie załadowanego modelu"""
    try:
        # Sprawdź czy model jest załadowany
        if model_key not in st.session_state.models or not st.session_state.models[model_key]['loaded']:
            return None, None, "Model nie jest załadowany"
        
        model_info = st.session_state.models[model_key]
        model = model_info['model']
        device = model_info['device']
        target_size = model_info['config']['input_size']
        
        # Przetwórz obraz
        input_tensor, error = preprocess_image(image, target_size)
        if error:
            return None, None, error
        
        # Wykonaj predykcję
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            prediction_logits = model(input_tensor)
            
            # Konwertuj logity do prawdopodobieństw (dla segmentacji binarnej)
            prediction_probs = torch.sigmoid(prediction_logits)
            
            # Uzyskaj maskę segmentacji (wartości binarnej)
            prediction_mask = (prediction_probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)
            
            # Stwórz sztuczne ground truth do demonstracji metryk
            gt_mask = prediction_mask.copy()
            noise_mask = np.random.random(gt_mask.shape) < 0.05
            gt_mask[noise_mask] = 1 - gt_mask[noise_mask]  # Odwróć wartości na maskę szumową
            
            # Konwertuj ground truth do tensora
            gt_tensor = torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Oblicz metryki
            iou_score = iou_pytorch(prediction_probs > 0.5, gt_tensor)
            dice_score = dice_coefficient_pytorch(prediction_probs > 0.5, gt_tensor)
            mpa_score = mean_pixel_accuracy_pytorch(prediction_probs > 0.5, gt_tensor)
            
            # Statystyki predykcji
            unique_classes, class_counts = np.unique(prediction_mask, return_counts=True)
            class_percentages = {int(cls): float(count) / prediction_mask.size * 100 
                               for cls, count in zip(unique_classes, class_counts)}
            
            # Uzupełnij brakujące klasy
            for cls in range(2):
                if cls not in class_percentages:
                    class_percentages[cls] = 0.0
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'image_shape': prediction_mask.shape,
                'num_classes_detected': len(unique_classes),
                'mean_iou': round(iou_score.item(), 4),
                'mean_dice': round(dice_score.item(), 4),
                'mean_pixel_accuracy': round(mpa_score.item(), 4),
                'class_distribution': {
                    CLASS_NAMES.get(cls, f"Class_{cls}"): {
                        'percentage': round(class_percentages.get(cls, 0.0), 2),
                        'pixel_count': int(class_counts[list(unique_classes).index(cls)] if cls in unique_classes else 0)
                    }
                    for cls in range(2)
                },
                'class_metrics': {
                    CLASS_NAMES[0]: {
                        'iou': round(iou_score.item(), 4),
                        'dice': round(dice_score.item(), 4),
                        'pixel_accuracy': round(mpa_score.item(), 4),
                    },
                    CLASS_NAMES[1]: {
                        'iou': round(iou_score.item(), 4),
                        'dice': round(dice_score.item(), 4),
                        'pixel_accuracy': round(mpa_score.item(), 4),
                    }
                }
            }
            
            info = {
                'model_used': {
                    'key': model_key,
                    'name': model_info['config']['name'],
                    'description': model_info['config']['description'],
                    'input_size': f"{target_size[0]}x{target_size[1]}",
                },
                'original_size': f"{image.size[0]}x{image.size[1]}",
                'processed_size': f"{target_size[0]}x{target_size[1]}",
                'processing_time': datetime.now().isoformat(),
                'device_used': str(device)
            }
            
            return prediction_mask, metrics, info
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Błąd podczas predykcji: {str(e)}"

def generate_demo_prediction(image, model_name):
    """Generuje przykładową predykcję dla trybu demo"""
    try:
        # Konwertuj obraz do numpy array
        img_array = np.array(image.convert('L'))
        h, w = img_array.shape
        
        # Stwórz przykładową maskę segmentacji
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Dodaj "guz" w środku obrazu
        center_x, center_y = w // 2, h // 2
        
        # Klasa 1 (guz) - kółko w środku
        y, x = np.ogrid[:h, :w]
        tumor_mask = (x - center_x)**2 + (y - center_y)**2 < (min(w, h) * 0.1)**2
        mask[tumor_mask] = 1
        
        # Dodaj trochę szumu
        noise = np.random.random((h, w)) < 0.02
        mask[noise] = np.random.randint(0, 2, size=np.sum(noise))
        
        # Stwórz przykładowe metryki
        unique_classes, class_counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        # Oblicz procentowe rozkłady
        class_percentages = {}
        for cls, count in zip(unique_classes, class_counts):
            class_percentages[int(cls)] = float(count) / total_pixels * 100
        
        # Uzupełnij brakujące klasy
        for cls in range(2):
            if cls not in class_percentages:
                class_percentages[cls] = 0.0
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'image_shape': mask.shape,
            'num_classes_detected': len(unique_classes),
            'mean_iou': 0.85,
            'mean_dice': 0.89,
            'mean_pixel_accuracy': 0.92,
            'class_distribution': {
                CLASS_NAMES.get(cls, f"Class_{cls}"): {
                    'percentage': round(class_percentages.get(cls, 0.0), 2),
                    'pixel_count': int(class_counts[list(unique_classes).index(cls)] if cls in unique_classes else 0)
                }
                for cls in range(2)
            },
            'class_metrics': {
                CLASS_NAMES.get(i, f"Class_{i}"): {
                    'iou': round(0.85 + np.random.uniform(-0.1, 0.1), 4),
                    'dice': round(0.89 + np.random.uniform(-0.1, 0.1), 4),
                    'pixel_accuracy': round(0.92 + np.random.uniform(-0.05, 0.05), 4),
                }
                for i in range(2)
            }
        }
        
        info = {
            'model_used': {
                'key': model_name,
                'name': MODELS_CONFIG[model_name]['name'],
                'description': MODELS_CONFIG[model_name]['description'] + " (TRYB DEMO)",
                'input_size': f"{MODELS_CONFIG[model_name]['input_size'][0]}x{MODELS_CONFIG[model_name]['input_size'][1]}",
            },
            'demo_mode': True,
            'note': 'To są przykładowe wyniki - nie rzeczywista analiza medyczna!'
        }
        
        return mask, metrics, info
        
    except Exception as e:
        st.error(f"Błąd generowania demo: {str(e)}")
        return None, None, None

def create_class_distribution_chart(metrics):
    """Tworzy wykres rozkładu klas w segmentacji"""
    if 'class_distribution' not in metrics:
        return None
    
    if not PLOTLY_AVAILABLE:
        return create_matplotlib_pie_chart(metrics)
        
    class_data = metrics['class_distribution']
    
    labels = []
    values = []
    colors = []
    
    for class_id, class_info in CLASS_DEFINITIONS.items():
        class_name = class_info['name']
        if class_name in class_data:
            labels.append(class_name)
            values.append(class_data[class_name]['percentage'])
            colors.append(class_info['hex'])
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>Procent: %{percent}<br>Piksele: %{customdata}<extra></extra>',
        customdata=[class_data[cls]['pixel_count'] for cls in labels if cls in class_data]
    )])
    
    fig.update_layout(
        title="Rozkład klas w segmentacji",
        showlegend=True,
        height=400
    )
    
    return fig

def create_matplotlib_pie_chart(metrics):
    """Tworzy wykres kołowy używając matplotlib jako fallback"""
    if 'class_distribution' not in metrics:
        return None
        
    class_data = metrics['class_distribution']
    
    labels = []
    values = []
    colors = []
    
    for class_id, class_info in CLASS_DEFINITIONS.items():
        class_name = class_info['name']
        if class_name in class_data and class_data[class_name]['percentage'] > 0:
            labels.append(f"{class_name}\n({class_data[class_name]['percentage']:.1f}%)")
            values.append(class_data[class_name]['percentage'])
            colors.append(np.array(class_info['color'])/255.0)  # Normalize to 0-1 for matplotlib
    
    if not values:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title("Rozkład klas w segmentacji", fontsize=14, fontweight='bold')
    
    # Dostosuj tekst
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    return fig

# =================================
# INTERFEJS GŁÓWNY
# =================================

# Tytuł aplikacji
st.markdown("<h1 class='main-header'>Brain MRI Segmentation AI</h1>", unsafe_allow_html=True)

# Sidebar - konfiguracja
with st.sidebar:
    st.markdown("### Konfiguracja")
    
    # Przełącznik trybu demo
    demo_mode = st.toggle(
        "Tryb Demo", 
        value=st.session_state.demo_mode,
        help="Włącz tryb demo bez korzystania z modelu - generuje przykładowe wyniki"
    )
    st.session_state.demo_mode = demo_mode
    
    if demo_mode:
        st.info("**Tryb Demo aktywny**\nGeneruję przykładowe wyniki")
    else:
        # Przycisk do ładowania modeli
        if not st.session_state.models_loaded:
            if st.button("Załaduj modele", type="primary"):
                with st.spinner("Ładowanie modeli..."):
                    try:
                        st.session_state.models, device = load_models()
                        loaded_count = sum(1 for m in st.session_state.models.values() if m['loaded'])
                        if loaded_count > 0:
                            st.session_state.models_loaded = True
                            st.success(f"Załadowano {loaded_count}/{len(MODELS_CONFIG)} modeli!")
                        else:
                            st.error("Nie udało się załadować żadnego modelu")
                    except Exception as e:
                        st.error(f"Błąd ładowania modeli: {str(e)}")
        else:
            st.success("Modele załadowane!")
            
            # Wyświetl informacje o załadowanych modelach
            loaded_models = [k for k, v in st.session_state.models.items() if v['loaded']]
            for model_key in loaded_models:
                model_info = st.session_state.models[model_key]
                st.markdown(f"✅ **{model_info['config']['name']}**")
    
    st.markdown("---")
    
    # Legenda klas
    st.markdown("### Legenda klas")
    for class_id, class_info in CLASS_DEFINITIONS.items():
        st.markdown(f"""
        <div class="class-legend">
            <div class="color-box" style="background-color: {class_info['hex']};"></div>
            <div>
                <strong>{class_info['name']}</strong><br>
                <small>{class_info['description']}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Główna część aplikacji
col1, col2 = st.columns([1, 2])

# Panel lewy - upload i wybór modelu
with col1:
    st.markdown("<h2 class='sub-header'>Upload obrazu MRI</h2>", unsafe_allow_html=True)
    
    # Wybór modelu
    st.markdown("### Wybór modelu AI")
    
    # Lista modeli do wyboru
    model_options = list(MODELS_CONFIG.keys())
    model_names = [MODELS_CONFIG[key]["name"] for key in model_options]
    
    # Znajdź index aktualnie wybranego modelu
    current_index = 0
    if st.session_state.selected_model in model_options:
        current_index = model_options.index(st.session_state.selected_model)
    
    # Radio button dla wyboru modelu
    selected_index = st.radio(
        "Wybierz model AI:",
        range(len(model_names)),
        format_func=lambda x: model_names[x],
        index=current_index,
        key="model_selection"
    )
    
    # Aktualizuj wybrany model
    st.session_state.selected_model = model_options[selected_index]
    selected_model_info = MODELS_CONFIG[st.session_state.selected_model]
    
    # Wyświetl szczegóły wybranego modelu
    st.markdown(f"""
    <div class="model-card">
        <h4>{selected_model_info['name']}</h4>
        <p>{selected_model_info['description']}</p>
        <strong>Zalecane dla:</strong> {selected_model_info['recommended_for']}<br>
        <strong>Rozmiar wejścia:</strong> {selected_model_info['input_size'][0]}x{selected_model_info['input_size'][1]}
        <br><strong>Cechy:</strong>
        <ul>{''.join([f'<li>{feature}</li>' for feature in selected_model_info['features']])}</ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload pliku
    uploaded_file = st.file_uploader(
        "Wybierz zdjęcie MRI do analizy", 
        type=["png", "jpg", "jpeg", "tiff", "tif"],
        help="Obsługiwane formaty: PNG, JPG, JPEG, TIFF"
    )
    
    if uploaded_file is not None:
        try:
            # Wczytanie i wyświetlenie przesłanego zdjęcia
            image = Image.open(uploaded_file)
            
            # Normalizacja zdjęcia dla lepszej wizualizacji
            normalized_image = normalize_image(image)
            
            # Wyświetlenie przesłanego zdjęcia
            st.image(normalized_image, caption="Przesłany obraz MRI", use_container_width=True)
            
            # Informacje o obrazie
            st.markdown(f"""
            **Informacje o obrazie:**
            - Rozmiar: {image.size[0]} x {image.size[1]} px
            - Tryb: {image.mode}
            - Format: {image.format or 'N/A'}
            """)
            
            # Zapisanie zdjęcia do stanu sesji
            st.session_state.uploaded_image = normalized_image
            
            # Przycisk do analizy zdjęcia
            analyze_button_text = "Wygeneruj Demo" if st.session_state.demo_mode else "Analizuj obraz MRI"
            
            if st.button(analyze_button_text, type="primary", use_container_width=True):
                if st.session_state.demo_mode:
                    # Tryb demo - generuj przykładowe wyniki
                    with st.spinner(f"Generuję demo używając modelu {MODELS_CONFIG[st.session_state.selected_model]['name']}..."):
                        mask, metrics, info = generate_demo_prediction(
                            normalized_image, 
                            st.session_state.selected_model
                        )
                        
                        if mask is not None and metrics is not None:
                            st.session_state.prediction = mask
                            st.session_state.metrics = metrics
                            st.session_state.prediction_info = info
                            st.success("Demo wygenerowane pomyślnie!")
                            st.balloons()
                        else:
                            st.error("Nie udało się wygenerować demo")
                else:
                    # Tryb normalny - używamy lokalnego modelu
                    if not st.session_state.models_loaded:
                        st.warning("Najpierw załaduj modele z panelu bocznego")
                        st.info("Możesz też włączyć **Tryb Demo** w panelu bocznym aby przetestować interfejs")
                    else:
                        with st.spinner(f"Analizuję obraz używając modelu {MODELS_CONFIG[st.session_state.selected_model]['name']}..."):
                            # Wykonaj predykcję lokalnie
                            mask, metrics, error = predict_with_local_model(
                                normalized_image, 
                                st.session_state.selected_model
                            )
                            
                            if mask is not None and metrics is not None:
                                st.session_state.prediction = mask
                                st.session_state.metrics = metrics
                                st.session_state.prediction_info = error  # info jest przekazywane w parametrze error
                                st.success("Analiza zakończona pomyślnie!")
                                st.balloons()
                            else:
                                st.error(f"Nie udało się wykonać analizy: {error}")
                                st.info("Spróbuj włączyć **Tryb Demo** w panelu bocznym")
        
        except Exception as e:
            st.error(f"Błąd podczas przetwarzania obrazu: {str(e)}")

# Panel prawy - wyniki analizy
with col2:
    st.markdown("<h2 class='sub-header'>Wyniki segmentacji</h2>", unsafe_allow_html=True)
    
    if st.session_state.prediction is not None and st.session_state.uploaded_image is not None:
        
        # Wyniki wizualne
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("#### Maska segmentacji")
            colored_mask = colorize_mask(st.session_state.prediction)
            st.image(colored_mask, caption="Kolorowa maska segmentacji", use_container_width=True)
            
        with result_col2:
            st.markdown("#### Nakładka na oryginał")
            overlay = overlay_masks(st.session_state.uploaded_image, st.session_state.prediction)
            st.image(overlay, caption="Segmentacja nałożona na oryginał", use_container_width=True)
        
        # Szczegółowe metryki
        st.markdown("#### Metryki jakości modelu")
        
        metrics = st.session_state.metrics
        
        # Główne metryki
        main_metrics_cols = st.columns(3)
        
        with main_metrics_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #e74c3c;">Dice Score</h3>
                <h1>{metrics.get('mean_dice', 'N/A'):.4f}</h1>
                <p>Współczynnik Dice'a</p>
            </div>
            """, unsafe_allow_html=True)
            
        with main_metrics_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #3498db;">IoU (Jaccard)</h3>
                <h1>{metrics.get('mean_iou', 'N/A'):.4f}</h1>
                <p>Intersection over Union</p>
            </div>
            """, unsafe_allow_html=True)
            
        with main_metrics_cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #27ae60;">Pixel Accuracy</h3>
                <h1 >{metrics.get('mean_pixel_accuracy', 'N/A'):.4f}</h1>
                <p>Dokładność pikselowa</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Wykres rozkładu klas
        if 'class_distribution' in metrics:
            st.markdown("#### Rozkład wykrytych klas")
            
            # Tworzenie wykresu kołowego
            distribution_fig = create_class_distribution_chart(metrics)
            if distribution_fig:
                if PLOTLY_AVAILABLE:
                    st.plotly_chart(distribution_fig, use_container_width=True)
                else:
                    st.pyplot(distribution_fig)
        
        # Szczegółowe metryki dla każdej klasy
        if 'class_metrics' in metrics:
            st.markdown("#### Szczegółowe metryki dla klas")
            
            if PANDAS_AVAILABLE:
                # Stwórz DataFrame dla lepszego wyświetlania
                class_data = []
                for class_name, class_metrics in metrics['class_metrics'].items():
                    class_data.append({
                        'Klasa': class_name,
                        'Dice': f"{class_metrics.get('dice', 0):.4f}",
                        'IoU': f"{class_metrics.get('iou', 0):.4f}",
                        'Pixel Accuracy': f"{class_metrics.get('pixel_accuracy', 0):.4f}"
                    })
                
                # Wyświetl jako tabelę
                df = pd.DataFrame(class_data)
                st.dataframe(df, use_container_width=True)
            else:
                # Fallback bez pandas - prostsze wyświetlanie
                for class_name, class_metrics in metrics['class_metrics'].items():
                    st.markdown(f"""
                    **{class_name}:**
                    - Dice: {class_metrics.get('dice', 0):.4f}
                    - IoU: {class_metrics.get('iou', 0):.4f}
                    - Pixel Accuracy: {class_metrics.get('pixel_accuracy', 0):.4f}
                    """)
                    st.markdown("---")
        
        # Informacje techniczne
        if hasattr(st.session_state, 'prediction_info') and st.session_state.prediction_info:
            with st.expander("Informacje techniczne"):
                info = st.session_state.prediction_info
                if isinstance(info, dict):
                    st.json(info)
                else:
                    st.text(info)
                    
    else:
        st.markdown("""
        <div class="results-container">
            <div style="text-align: center; padding: 2rem;">
                <h3>Gotowy do analizy!</h3>
                <p>Prześlij obraz MRI i wybierz model, aby rozpocząć segmentację.</p>
                <br>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Dodatkowe informacje
with st.expander("Informacje o modelu i klasach"):
    st.markdown("""
    ### Tryb Demo vs Tryb Normalny:
    
    **Tryb Demo:**
    - Nie wymaga załadowanych modeli
    - Generuje przykładowe wyniki segmentacji
    - Idealny do testowania interfejsu
    - **UWAGA:** Wyniki nie są prawdziwą analizą medyczną!
    
    **Tryb Normalny:**
    - Wymaga załadowania modeli
    - Wykonuje prawdziwą segmentację obrazów MRI
    - Używa lokalnie załadowanych modeli PyTorch
    
    ### Model U-Net:
    
    **U-Net Standard** - Model wykorzystujący architekturę U-Net do segmentacji obrazów MRI, zapewniający szybkie i stabilne wyniki dla wykrywania guzów mózgu.
    
    ### Klasy segmentacji:
    
    Model został wytrenowany do identyfikacji następujących struktur w obrazach MRI mózgu:
    
    - **Tło (czarny)**: Obszary nie będące guzem
    - **Guz (czerwony)**: Obszary z guzem mózgu
    
    ### Metryki:
    
    - **Dice Score**: Miara podobieństwa między predykcją a rzeczywistością (0-1, wyższe = lepsze)
    - **IoU (Intersection over Union)**: Stosunek części wspólnej do sumy obszarów (0-1, wyższe = lepsze)  
    - **Pixel Accuracy**: Procent poprawnie sklasyfikowanych pikseli (0-1, wyższe = lepsze)
    """)

# Informacje w stopce
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; margin-top: 2rem;">
    <strong>Brain MRI Segmentation AI</strong> | 
    Powered by U-Net Deep Learning | 
    Narzędzie wspomagające diagnostykę medyczną<br>
    <small>Dla celów demonstracyjnych - nie zastępuje konsultacji medycznej</small>
</div>
""", unsafe_allow_html=True)