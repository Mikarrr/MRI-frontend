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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfiguracja strony
st.set_page_config(
    layout="wide", 
    page_title="🧠 Brain MRI Segmentation AI",
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
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
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
        background: linear-gradient(135deg, #ffffff 0%, #f1f3f4 100%);
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
        background: #f8f9fa;
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
</style>
""", unsafe_allow_html=True)

# Definicje modeli
MODELS_CONFIG = {
    "unet_standard": {
        "name": "🎯 U-Net Standard",
        "description": "Podstawowy model U-Net z standardowymi parametrami",
        "checkpoint": "best_unet_model.pth",
        "input_size": (256, 256),
        "features": ["Szybka predykcja", "Dobra ogólna jakość", "Stabilny"],
        "recommended_for": "Ogólne zastosowania diagnostyczne"
    },
    "unet_enhanced": {
        "name": "⚡ U-Net Enhanced", 
        "description": "Ulepszona wersja z większą liczbą filtrów i attention",
        "checkpoint": "best_unet_enhanced.pth",
        "input_size": (256, 256),
        "features": ["Wyższa dokładność", "Lepsze wykrywanie detali", "Attention mechanism"],
        "recommended_for": "Precyzyjna analiza zmian patologicznych"
    },
    "unet_deep": {
        "name": "🔬 U-Net Deep",
        "description": "Głęboka architektura dla najwyższej precyzji",
        "checkpoint": "best_unet_deep.pth", 
        "input_size": (512, 512),
        "features": ["Najwyższa dokładność", "Wysoka rozdzielczość", "Zaawansowana architektura"],
        "recommended_for": "Badania naukowe i przypadki skomplikowane"
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

# Tytuł aplikacji
st.markdown("<h1 class='main-header'>🧠 Brain MRI Segmentation AI</h1>", unsafe_allow_html=True)

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

# Funkcja do tworzenia wykresu rozkładu klas
def create_class_distribution_chart(metrics):
    """Tworzy wykres rozkładu klas w segmentacji"""
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

# === GŁÓWNY INTERFEJS ===

# Sidebar - konfiguracja
with st.sidebar:
    st.markdown("### ⚙️ Konfiguracja")
    
    # URL serwera
    server_url = st.text_input(
        "🌐 URL serwera Flask:",
        value="http://localhost:5000",
        help="Adres serwera z uruchomionym modelem"
    )
    
    # Sprawdzenie statusu serwera
    if st.button("🔄 Sprawdź status serwera"):
        with st.spinner("Sprawdzam serwer..."):
            status, info = check_server_status(server_url)
            st.session_state.server_status = status
            
            if status == "online":
                st.success("✅ Serwer jest dostępny!")
                if info:
                    st.json(info)
            else:
                st.error("❌ Serwer niedostępny")
    
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
    st.markdown("### 🎨 Legenda klas")
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
    st.markdown("<h2 class='sub-header'>📤 Upload obrazu MRI</h2>", unsafe_allow_html=True)
    
    # Wybór modelu
    st.markdown("### 🤖 Wybór modelu AI")
    
    for model_key, model_info in MODELS_CONFIG.items():
        is_selected = st.session_state.selected_model == model_key
        
        # Klasa CSS w zależności od wyboru
        card_style = "model-card" + (" selected-model" if is_selected else "")
        
        with st.container():
            if st.radio(
                "Wybierz model:",
                [model_key],
                format_func=lambda x: MODELS_CONFIG[x]["name"],
                key=f"radio_{model_key}",
                index=0 if is_selected else -1,
                label_visibility="collapsed"
            ):
                st.session_state.selected_model = model_key
            
            # Szczegóły modelu
            if is_selected:
                st.markdown(f"""
                <div class="model-card">
                    <h4>{model_info['name']}</h4>
                    <p>{model_info['description']}</p>
                    <strong>Zalecane dla:</strong> {model_info['recommended_for']}<br>
                    <strong>Rozmiar wejścia:</strong> {model_info['input_size'][0]}x{model_info['input_size'][1]}
                    <br><strong>Cechy:</strong>
                    <ul>{''.join([f'<li>{feature}</li>' for feature in model_info['features']])}</ul>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload pliku
    uploaded_file = st.file_uploader(
        "📁 Wybierz zdjęcie MRI do analizy", 
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
            st.image(normalized_image, caption="📸 Przesłany obraz MRI", use_container_width=True)
            
            # Informacje o obrazie
            st.markdown(f"""
            **📊 Informacje o obrazie:**
            - Rozmiar: {image.size[0]} x {image.size[1]} px
            - Tryb: {image.mode}
            - Format: {image.format or 'N/A'}
            """)
            
            # Zapisanie zdjęcia do stanu sesji
            st.session_state.uploaded_image = normalized_image
            
            # Przycisk do analizy zdjęcia
            if st.button("🔍 Analizuj obraz MRI", type="primary", use_container_width=True):
                if st.session_state.server_status != "online":
                    st.warning("⚠️ Sprawdź czy serwer jest dostępny przed analizą")
                else:
                    with st.spinner(f"🧠 Analizuję obraz używając modelu {MODELS_CONFIG[st.session_state.selected_model]['name']}..."):
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
                            st.success("✅ Analiza zakończona pomyślnie!")
                            st.balloons()
                        else:
                            st.error("❌ Nie udało się wykonać analizy")
        
        except Exception as e:
            st.error(f"❌ Błąd podczas przetwarzania obrazu: {str(e)}")

# Panel prawy - wyniki analizy
with col2:
    st.markdown("<h2 class='sub-header'>📊 Wyniki segmentacji</h2>", unsafe_allow_html=True)
    
    if st.session_state.prediction is not None and st.session_state.uploaded_image is not None:
        # Wyniki wizualne
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("#### 🎨 Maska segmentacji")
            colored_mask = colorize_mask(st.session_state.prediction)
            st.image(colored_mask, caption="Kolorowa maska segmentacji", use_container_width=True)
            
        with result_col2:
            st.markdown("#### 🖼️ Nakładka na oryginał")
            overlay = overlay_masks(st.session_state.uploaded_image, st.session_state.prediction)
            st.image(overlay, caption="Segmentacja nałożona na oryginał", use_container_width=True)
        
        # Szczegółowe metryki
        st.markdown("#### 📈 Metryki jakości modelu")
        
        metrics = st.session_state.metrics
        
        # Główne metryki
        main_metrics_cols = st.columns(3)
        
        with main_metrics_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #e74c3c;">🎯 Dice Score</h3>
                <h1 style="color: #2c3e50;">{metrics.get('mean_dice', 'N/A'):.4f}</h1>
                <p>Średnia ze wszystkich klas</p>
            </div>
            """, unsafe_allow_html=True)
            
        with main_metrics_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #3498db;">📐 IoU (Jaccard)</h3>
                <h1 style="color: #2c3e50;">{metrics.get('mean_iou', 'N/A'):.4f}</h1>
                <p>Intersection over Union</p>
            </div>
            """, unsafe_allow_html=True)
            
        with main_metrics_cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #27ae60;">✅ Pixel Accuracy</h3>
                <h1 style="color: #2c3e50;">{metrics.get('mean_pixel_accuracy', 'N/A'):.4f}</h1>
                <p>Dokładność pikselowa</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Wykres rozkładu klas
        if 'class_distribution' in metrics:
            st.markdown("#### 📊 Rozkład wykrytych klas")
            
            # Tworzenie wykresu kołowego
            distribution_fig = create_class_distribution_chart(metrics)
            if distribution_fig:
                st.plotly_chart(distribution_fig, use_container_width=True)
        
        # Szczegółowe metryki dla każdej klasy
        if 'class_metrics' in metrics:
            st.markdown("#### 🔍 Szczegółowe metryki dla klas")
            
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
            import pandas as pd
            df = pd.DataFrame(class_data)
            st.dataframe(df, use_container_width=True)
        
        # Informacje techniczne
        if hasattr(st.session_state, 'prediction_info') and st.session_state.prediction_info:
            with st.expander("🔧 Informacje techniczne"):
                info = st.session_state.prediction_info
                st.json(info)
                    
    else:
        st.markdown("""
        <div class="results-container">
            <div style="text-align: center; padding: 2rem;">
                <h3>🏥 Gotowy do analizy!</h3>
                <p>Prześlij obraz MRI i wybierz model, aby rozpocząć segmentację.</p>
                <br>
                <h4>📋 Kroki:</h4>
                <ol style="text-align: left; display: inline-block;">
                    <li>Sprawdź status serwera w panelu bocznym</li>
                    <li>Wybierz odpowiedni model AI</li>
                    <li>Prześlij obraz MRI</li>
                    <li>Kliknij "Analizuj obraz MRI"</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Dodatkowe informacje
with st.expander("📚 Informacje o modelach i klasach"):
    st.markdown("""
    ### 🤖 Dostępne modele:
    
    **🎯 U-Net Standard** - Podstawowy model zapewniający szybkie i stabilne wyniki dla większości przypadków diagnostycznych.
    
    **⚡ U-Net Enhanced** - Ulepszona wersja z mechanizmami uwagi, zapewniająca wyższą dokładność wykrywania szczegółów patologicznych.
    
    **🔬 U-Net Deep** - Najzaawansowana architektura dla przypadków wymagających najwyższej precyzji i analizy wysokiej rozdzielczości.
    
    ### 🧠 Klasy segmentacji:
    
    Model został wytrenowany do identyfikacji następujących struktur w obrazach MRI mózgu:
    
    - **Tło (czarny)**: Obszary nie będące tkanką mózgową
    - **Nekrotyczny rdzeń (czerwony)**: NCR/NET - obszary nekrotyczne i nie wzmacniające się części guza
    - **Obrzęk okołoguzowy (zielony)**: ED - obrzęk wokół guza
    - **Aktywny guz (niebieski)**: ET - aktywnie wzmacniające się części guza
    
    ### 📊 Metryki:
    
    - **Dice Score**: Miara podobieństwa między predykcją a rzeczywistością (0-1, wyższe = lepsze)
    - **IoU (Intersection over Union)**: Stosunek części wspólnej do sumy obszarów (0-1, wyższe = lepsze)  
    - **Pixel Accuracy**: Procent poprawnie sklasyfikowanych pikseli (0-1, wyższe = lepsze)
    """)

# Informacje w stopce
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; margin-top: 2rem;">
    🧠 <strong>Brain MRI Segmentation AI</strong> | 
    Powered by U-Net Deep Learning | 
    🔬 Narzędzie wspomagające diagnostykę medyczną
</div>
""", unsafe_allow_html=True)