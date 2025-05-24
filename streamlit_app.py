import streamlit as st
import torch
import numpy as np
import io
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import requests
import tempfile
import os
import json
from urllib.parse import urljoin
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
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
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

# Definicje modeli - TUTAJ DODAJ SWOJE MODELE
MODELS_CONFIG = {
    "twoj_model_1": {
        "name": "Twój Model 1",
        "description": "Opis Twojego pierwszego modelu",
        "checkpoint": "sciezka/do/twojego/modelu1.pth",  # Ścieżka do pliku modelu
        "input_size": (256, 256),  # Rozmiar wejściowy obrazu (szerokość, wysokość)
        "features": ["Cecha 1", "Cecha 2", "Cecha 3"],  # Lista cech modelu
        "recommended_for": "Do czego najlepiej nadaje się Twój model"
    },
    "twoj_model_2": {
        "name": "Twój Model 2", 
        "description": "Opis Twojego drugiego modelu",
        "checkpoint": "sciezka/do/twojego/modelu2.pth",
        "input_size": (512, 512),
        "features": ["Inna cecha 1", "Inna cecha 2"],
        "recommended_for": "Inne zastosowanie"
    },
    # Dodaj więcej modeli według potrzeb:
    "unet_standard": {
        "name": "U-Net Standard",
        "description": "Podstawowy model U-Net z standardowymi parametrami",
        "checkpoint": "best_unet_model.pth",
        "input_size": (256, 256),
        "features": ["Szybka predykcja", "Dobra ogólna jakość", "Stabilny"],
        "recommended_for": "Ogólne zastosowania diagnostyczne"
    }
}

# Definicje klas segmentacji - zgodne z Twoim modelem
CLASS_DEFINITIONS = {
    0: {
        "name": "Tło",
        "description": "Obszary nie będące tkanką mózgową",
        "color": [0, 0, 0],
        "hex": "#000000"
    },
    1: {
        "name": "Nekrotyczny rdzeń",
        "description": "Nekrotyczny rdzeń guza (NCR/NET)",
        "color": [255, 0, 0],
        "hex": "#FF0000"
    },
    2: {
        "name": "Obrzęk okołoguzowy", 
        "description": "Obrzęk wokół guza (ED)",
        "color": [0, 255, 0],
        "hex": "#00FF00"
    },
    3: {
        "name": "Aktywny guz",
        "description": "Aktywne tkanka guza (ET)",
        "color": [0, 0, 255],
        "hex": "#0000FF"
    }
}

# CLASS_NAMES dla kompatybilności
CLASS_NAMES = {i: info["name"] for i, info in CLASS_DEFINITIONS.items()}

# Inicjalizacja sesji
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "unet_standard"
if 'server_status' not in st.session_state:
    st.session_state.server_status = "unknown"
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# Tytuł aplikacji
st.markdown("<h1 class='main-header'>Brain MRI Segmentation AI</h1>", unsafe_allow_html=True)

# Informacja o brakujących bibliotekach
missing_libs = []
if not PLOTLY_AVAILABLE:
    missing_libs.append("plotly")
if not PANDAS_AVAILABLE:
    missing_libs.append("pandas")

if missing_libs:
    st.info(f"""
    Opcjonalne biblioteki: Dla pełnej funkcjonalności zainstaluj brakujące biblioteki:
    ```bash
    pip install {' '.join(missing_libs)}
    ```
    Aplikacja będzie działać w trybie podstawowym bez nich.
    """)

# Funkcja sprawdzania statusu serwera
def check_server_status(server_url):
    """Sprawdza czy serwer Flask jest dostępny"""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            return "online", response.json()
        else:
            return "offline", None
    except Exception as e:
        return "offline", str(e)

# Funkcja normalizująca obraz
def normalize_image(img):
    """Normalizacja obrazu MRI dla lepszej wizualizacji"""
    img_np = np.array(img.convert('L'))
    img_min, img_max = img_np.min(), img_np.max()
    if img_max > img_min:
        img_np = ((img_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_np = np.zeros_like(img_np, dtype=np.uint8)
    return Image.fromarray(img_np)

# Funkcja do generowania kolorowej maski
def colorize_mask(mask):
    """Generuje kolorową maskę na podstawie predykcji"""
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, class_info in CLASS_DEFINITIONS.items():
        rgb_mask[mask == class_id] = class_info["color"]
    return rgb_mask

# Funkcja połączenia zdjęcia z maską z odpowiednią przezroczystością
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

# Funkcja do predykcji z serwerem Flask
def predict_with_flask_server(image, server_url, model_name):
    """Wysyła obraz do serwera Flask i otrzymuje predykcję"""
    try:
        # Konweruj obraz do bajtów
        img_bytes = io.BytesIO()
        # Upewnij się, że obraz jest w trybie RGB lub L
        if image.mode not in ['RGB', 'L']:
            image = image.convert('L')
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Przygotowanie pliku do wysłania
        files = {'file': ('brain_scan.png', img_bytes, 'image/png')}
        data = {'model': model_name}  # Wyślij informację o wybranym modelu
        
        # Wysłanie żądania do serwera Flask
        response = requests.post(f"{server_url}/predict", files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                prediction_mask = np.array(result['segmentation_mask'])
                metrics = result['metrics']
                info = result.get('info', {})
                return prediction_mask, metrics, info
            else:
                st.error(f"Błąd predykcji: {result.get('error', 'Nieznany błąd')}")
                return None, None, None
        else:
            st.error(f"Błąd serwera: {response.status_code}")
            st.error(response.text)
            return None, None, None
            
    except requests.exceptions.Timeout:
        st.error("Przekroczono limit czasu - serwer nie odpowiada")
        return None, None, None
    except Exception as e:
        st.error(f"Wystąpił błąd: {str(e)}")
        return None, None, None

# Funkcja do generowania demo danych
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
        
        # Klasa 3 (aktywny guz) - małe kółko w środku
        y, x = np.ogrid[:h, :w]
        tumor_mask = (x - center_x)**2 + (y - center_y)**2 < (min(w, h) * 0.05)**2
        mask[tumor_mask] = 3
        
        # Klasa 1 (nekrotyczny rdzeń) - wokół guza
        necrotic_mask = (x - center_x)**2 + (y - center_y)**2 < (min(w, h) * 0.08)**2
        necrotic_mask = necrotic_mask & ~tumor_mask
        mask[necrotic_mask] = 1
        
        # Klasa 2 (obrzęk) - większy obszar wokół
        edema_mask = (x - center_x)**2 + (y - center_y)**2 < (min(w, h) * 0.12)**2
        edema_mask = edema_mask & ~necrotic_mask & ~tumor_mask
        mask[edema_mask] = 2
        
        # Dodaj trochę szumu
        noise = np.random.random((h, w)) < 0.02
        mask[noise] = np.random.randint(0, 4, size=np.sum(noise))
        
        # Stwórz przykładowe metryki
        unique_classes, class_counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        # Oblicz procentowe rozkłady
        class_percentages = {}
        for cls, count in zip(unique_classes, class_counts):
            class_percentages[int(cls)] = float(count) / total_pixels * 100
        
        # Przykładowe metryki (różne dla różnych modeli)
        model_quality = {
            "unet_standard": {"dice": 0.78, "iou": 0.68, "accuracy": 0.88},
            "unet_enhanced": {"dice": 0.84, "iou": 0.76, "accuracy": 0.92}, 
            "unet_deep": {"dice": 0.89, "iou": 0.82, "accuracy": 0.95}
        }
        
        base_metrics = model_quality.get(model_name, model_quality["unet_standard"])
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'image_shape': mask.shape,
            'num_classes_detected': len(unique_classes),
            'mean_iou': base_metrics["iou"],
            'mean_dice': base_metrics["dice"],
            'mean_pixel_accuracy': base_metrics["accuracy"],
            'class_distribution': {
                CLASS_NAMES.get(cls, f"Class_{cls}"): {
                    'percentage': round(class_percentages.get(cls, 0.0), 2),
                    'pixel_count': int(class_counts[list(unique_classes).index(cls)] if cls in unique_classes else 0)
                }
                for cls in range(4)
            },
            'class_metrics': {
                CLASS_NAMES.get(i, f"Class_{i}"): {
                    'iou': round(base_metrics["iou"] + np.random.uniform(-0.1, 0.1), 4),
                    'dice': round(base_metrics["dice"] + np.random.uniform(-0.1, 0.1), 4),
                    'pixel_accuracy': round(base_metrics["accuracy"] + np.random.uniform(-0.05, 0.05), 4),
                }
                for i in range(4)
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

# Funkcja do tworzenia wykresu rozkładu klas
def create_class_distribution_chart(metrics):
    """Tworzy wykres liniowy rozkładu klas w segmentacji"""
    if 'class_distribution' not in metrics:
        return None
    
    if not PLOTLY_AVAILABLE:
        return create_matplotlib_line_chart(metrics)
        
    class_data = metrics['class_distribution']
    
    labels = []
    values = []
    colors = []
    pixel_counts = []
    
    for class_id, class_info in CLASS_DEFINITIONS.items():
        class_name = class_info['name']
        if class_name in class_data:
            labels.append(class_name)
            values.append(class_data[class_name]['percentage'])
            colors.append(class_info['hex'])
            pixel_counts.append(class_data[class_name]['pixel_count'])
    
    fig = go.Figure()
    
    # Dodaj wykres liniowy
    fig.add_trace(go.Scatter(
        x=labels,
        y=values,
        mode='lines+markers',
        name='Rozkład klas (%)',
        line=dict(color='#3498db', width=3),
        marker=dict(
            size=12,
            color=colors,
            line=dict(color='white', width=2)
        ),
        hovertemplate='<b>%{x}</b><br>Procent: %{y:.2f}%<br>Piksele: %{customdata}<extra></extra>',
        customdata=pixel_counts
    ))
    
    # Dodaj wypełnienie pod linią
    fig.add_trace(go.Scatter(
        x=labels,
        y=values,
        fill='tonexty',
        mode='none',
        fillcolor='rgba(52, 152, 219, 0.2)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title="Rozkład klas w segmentacji",
        xaxis_title="Klasy segmentacji",
        yaxis_title="Procent pokrycia (%)",
        showlegend=True,
        height=400,
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            tickangle=45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            range=[0, max(values) * 1.1] if values else [0, 100]
        )
    )
    
    return fig

def create_matplotlib_line_chart(metrics):
    """Tworzy wykres liniowy używając matplotlib jako fallback"""
    if 'class_distribution' not in metrics:
        return None
        
    class_data = metrics['class_distribution']
    
    labels = []
    values = []
    colors = []
    
    for class_id, class_info in CLASS_DEFINITIONS.items():
        class_name = class_info['name']
        if class_name in class_data:
            labels.append(class_name)
            values.append(class_data[class_name]['percentage'])
            colors.append(np.array(class_info['color'])/255.0)  # Normalize to 0-1 for matplotlib
    
    if not values:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Wykres liniowy z markerami
    line = ax.plot(labels, values, 'o-', linewidth=3, markersize=8, color='#3498db')
    
    # Wypełnienie pod linią
    ax.fill_between(labels, values, alpha=0.3, color='#3498db')
    
    # Kolorowe markery dla każdej klasy
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        ax.scatter(i, value, color=color, s=100, edgecolor='white', linewidth=2, zorder=5)
        # Dodaj etykiety z wartościami
        ax.annotate(f'{value:.1f}%', (i, value), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontweight='bold')
    
    ax.set_title("Rozkład klas w segmentacji", fontsize=14, fontweight='bold')
    ax.set_xlabel("Klasy segmentacji", fontsize=12)
    ax.set_ylabel("Procent pokrycia (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(values) * 1.2 if values else 100)
    
    # Obróć etykiety osi X
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

# === GŁÓWNY INTERFEJS ===

# Sidebar - konfiguracja
with st.sidebar:
    st.markdown("### Konfiguracja")
    
    # Przełącznik trybu demo
    demo_mode = st.toggle(
        "Tryb Demo", 
        value=st.session_state.demo_mode,
        help="Włącz tryb demo bez potrzeby serwera Flask - generuje przykładowe wyniki"
    )
    st.session_state.demo_mode = demo_mode
    
    if demo_mode:
        st.info("**Tryb Demo aktywny**\nGeneruję przykładowe wyniki bez serwera")
    else:
        # URL serwera (tylko jeśli nie demo)
        server_url = st.text_input(
            "URL serwera Flask:",
            value="http://localhost:5000",
            help="Adres serwera z uruchomionym modelem"
        )
        
        # Sprawdzenie statusu serwera
        if st.button("Sprawdź status serwera"):
            with st.spinner("Sprawdzam serwer..."):
                status, info = check_server_status(server_url)
                st.session_state.server_status = status
                
                if status == "online":
                    st.success("Serwer jest dostępny!")
                    if info:
                        st.json(info)
                else:
                    st.error("Serwer niedostępny")
        
        # Wyświetl aktualny status
        status_color = {
            "online": "status-online",
            "offline": "status-offline", 
            "unknown": "status-loading"
        }
        
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <span class="status-indicator {status_color.get(st.session_state.server_status, 'status-loading')}"></span>
            Status serwera: <strong>{st.session_state.server_status.upper()}</strong>
        </div>
        """, unsafe_allow_html=True)
    
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
    
    # Wybór modelu (uproszczona wersja)
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
                    # Tryb normalny - połączenie z serwerem
                    if st.session_state.server_status != "online":
                        st.warning("Sprawdź czy serwer jest dostępny przed analizą")
                        st.info("Możesz włączyć **Tryb Demo** w panelu bocznym aby przetestować interfejs")
                    else:
                        with st.spinner(f"Analizuję obraz używając modelu {MODELS_CONFIG[st.session_state.selected_model]['name']}..."):
                            # Wywołanie predykcji z serwerem Flask
                            mask, metrics, info = predict_with_flask_server(
                                normalized_image, 
                                server_url, 
                                st.session_state.selected_model
                            )
                            
                            if mask is not None and metrics is not None:
                                st.session_state.prediction = mask
                                st.session_state.metrics = metrics
                                st.session_state.prediction_info = info
                                st.success("Analiza zakończona pomyślnie!")
                                st.balloons()
                            else:
                                st.error("Nie udało się wykonać analizy")
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
                <h1 style="color: #2c3e50;">{metrics.get('mean_dice', 'N/A'):.4f}</h1>
                <p>Średnia ze wszystkich klas</p>
            </div>
            """, unsafe_allow_html=True)
            
        with main_metrics_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #3498db;">IoU (Jaccard)</h3>
                <h1 style="color: #2c3e50;">{metrics.get('mean_iou', 'N/A'):.4f}</h1>
                <p>Intersection over Union</p>
            </div>
            """, unsafe_allow_html=True)
            
        with main_metrics_cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #27ae60;">Pixel Accuracy</h3>
                <h1 style="color: #2c3e50;">{metrics.get('mean_pixel_accuracy', 'N/A'):.4f}</h1>
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
                        'MPA': f"{class_metrics.get('pixel_accuracy', 0):.4f}"
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
                    - MPA: {class_metrics.get('pixel_accuracy', 0):.4f}
                    """)
                    st.markdown("---")
        
        # Informacje techniczne
        if hasattr(st.session_state, 'prediction_info') and st.session_state.prediction_info:
            with st.expander("Informacje techniczne"):
                info = st.session_state.prediction_info
                st.json(info)
                    
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
with st.expander("Informacje o modelach i klasach"):
    st.markdown("""
    ### Tryb Demo vs Tryb Rzeczywisty:
    
    **Tryb Demo:**
    - Nie wymaga serwera Flask ani wytrenowanych modeli
    - Generuje przykładowe wyniki segmentacji
    - Idealny do testowania interfejsu
    - **UWAGA:** Wyniki nie są prawdziwą analizą medyczną!
    
    **Tryb Rzeczywisty:**
    - Wymaga uruchomionego serwera Flask z wytrenowanymi modelami
    - Wykonuje prawdziwą segmentację obrazów MRI
    - Różne poziomy dokładności w zależności od modelu
    
    ### Dostępne modele:
    
    **U-Net Standard** - Podstawowy model zapewniający szybkie i stabilne wyniki dla większości przypadków diagnostycznych.
    
    **U-Net Enhanced** - Ulepszona wersja z mechanizmami uwagi, zapewniająca wyższą dokładność wykrywania szczegółów patologicznych.
    
    **U-Net Deep** - Najzaawansowana architektura dla przypadków wymagających najwyższej precyzji i analizy wysokiej rozdzielczości.
    
    ### Klasy segmentacji:
    
    Model został wytrenowany do identyfikacji następujących struktur w obrazach MRI mózgu:
    
    - **Tło (czarny)**: Obszary nie będące tkanką mózgową
    - **Nekrotyczny rdzeń (czerwony)**: NCR/NET - obszary nekrotyczne i nie wzmacniające się części guza
    - **Obrzęk okołoguzowy (zielony)**: ED - obrzęk wokół guza
    - **Aktywny guz (niebieski)**: ET - aktywnie wzmacniające się części guza
    
    ### Metryki:
    
    - **Dice Score**: Miara podobieństwa między predykcją a rzeczywistością (0-1, wyższe = lepsze)
    - **IoU (Intersection over Union)**: Stosunek części wspólnej do sumy obszarów (0-1, wyższe = lepsze)  
    - **Valid MPA (Mean Pixel Accuracy)**: Średnia dokładność pikselowa ze wszystkich klas (0-1, wyższe = lepsze)
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