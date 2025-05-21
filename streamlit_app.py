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

# Konfiguracja strony
st.set_page_config(layout="wide", page_title="Brain MRI Segmentation")

# Style CSS dla interfejsu
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #2c3e50;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .results-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.25rem 0.5rem rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.3rem;
        box-shadow: 0 0.1rem 0.3rem rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Tytuł aplikacji
st.markdown("<h1 class='main-header'>Brain MRI Segmentation AI</h1>", unsafe_allow_html=True)

# Inicjalizacja sesji
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# Funkcja normalizująca obraz
def normalize_image(img):
    img_np = np.array(img.convert('L'))
    img_min, img_max = img_np.min(), img_np.max()
    if img_max > img_min:
        img_np = ((img_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_np = np.zeros_like(img_np, dtype=np.uint8)
    return Image.fromarray(img_np)

# Funkcja do przetwarzania obrazu dla modelu
def preprocess_image(img, size=(256, 256)):
    # Upewnij się, że obraz jest w odcieniach szarości
    img = img.convert('L')
    img = img.resize(size, Image.BILINEAR)
    
    # Konwersja do tensora i normalizacja
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0)  # Dodaj wymiar batch_size

# Funkcja do generowania kolorowej maski
def colorize_mask(mask):
    # Kolory dla poszczególnych klas (tło, klasa 1, klasa 2, klasa 3)
    colors = [
        [0, 0, 0],        # Tło - czarny
        [255, 0, 0],      # Klasa 1 - czerwony
        [0, 255, 0],      # Klasa 2 - zielony
        [0, 0, 255]       # Klasa 3 - niebieski
    ]
    
    # Konwersja maski na obraz RGB
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(len(colors)):
        rgb_mask[mask == i] = colors[i]
    
    return rgb_mask

# Funkcja połączenia zdjęcia z maską z odpowiednią przezroczystością
def overlay_masks(image, mask, alpha=0.5):
    image = np.array(image.convert('RGB'))
    
    # Upewnij się, że oba obrazy mają ten sam rozmiar
    if image.shape[:2] != mask.shape[:2]:
        mask = Image.fromarray(mask)
        mask = mask.resize((image.shape[1], image.shape[0]), Image.NEAREST)
        mask = np.array(mask)
    
    # Nakładka maski na obraz
    blended = image.copy()
    colored_mask = colorize_mask(mask)
    mask_indices = mask > 0  # Indeksy, gdzie maska ma wartości
    
    # Nakładka tylko tam, gdzie maska ma wartości
    blended_pixels = mask_indices[:, :, np.newaxis].repeat(3, axis=2)
    blended[blended_pixels] = (alpha * colored_mask[blended_pixels] + 
                             (1 - alpha) * image[blended_pixels]).astype(np.uint8)
    
    return blended

# Funkcja do połączenia z Google Colab i predykcji
def predict_from_colab(img_tensor, colab_url):
    try:
        # Konwersja tensora na bajty
        img_bytes = io.BytesIO()
        img_pil = transforms.ToPILImage()(img_tensor.squeeze(0))
        img_pil.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Przygotowanie pliku do wysłania
        files = {'file': ('image.png', img_bytes, 'image/png')}
        
        # Wysłanie żądania do Google Colab
        response = requests.post(colab_url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            prediction_mask = np.array(result['mask'])
            metrics = result['metrics']
            return prediction_mask, metrics
        else:
            st.error(f"Błąd połączenia z Colab: {response.status_code}")
            st.error(response.text)
            return None, None
            
    except Exception as e:
        st.error(f"Wystąpił błąd: {str(e)}")
        return None, None

# Utworzenie dwukolumnowego układu
col1, col2 = st.columns([1, 2])

# Panel lewy - formularz do przesłania zdjęcia
with col1:
    st.markdown("<h2 class='sub-header'>Prześlij zdjęcie MRI</h2>", unsafe_allow_html=True)
    
    # Pole do przesłania pliku
    uploaded_file = st.file_uploader("Wybierz zdjęcie MRI do analizy", type=["png", "jpg", "jpeg"])
    
    # Pole na URL do Google Colab
    colab_url = st.text_input(
        "URL do Google Colab (np. https://xxx.ngrok.io/predict)",
        value="https://example.ngrok.io/predict"
    )
    
    if uploaded_file is not None:
        try:
            # Wczytanie i wyświetlenie przesłanego zdjęcia
            image = Image.open(uploaded_file)
            
            # Normalizacja zdjęcia (przyda się przy zdjęciach MRI)
            image = normalize_image(image)
            
            # Wyświetlenie przesłanego zdjęcia
            st.image(image, caption="Przesłane zdjęcie", use_container_width=True)
            
            # Zapisanie zdjęcia do stanu sesji
            st.session_state.uploaded_image = image
            
            # Przycisk do analizy zdjęcia
            if st.button("Analizuj zdjęcie"):
                with st.spinner("Analizuję zdjęcie..."):
                    # Przetworzenie zdjęcia dla modelu
                    img_tensor = preprocess_image(image)
                    
                    # Wywołanie predykcji z Google Colab
                    mask, metrics = predict_from_colab(img_tensor, colab_url)
                    
                    if mask is not None and metrics is not None:
                        st.session_state.prediction = mask
                        st.session_state.metrics = metrics
                        st.success("Analiza zakończona!")
                    else:
                        st.error("Nie udało się wykonać analizy. Sprawdź połączenie z Google Colab.")
        
        except Exception as e:
            st.error(f"Błąd podczas przetwarzania zdjęcia: {str(e)}")

# Panel prawy - wyniki analizy
with col2:
    st.markdown("<h2 class='sub-header'>Wyniki segmentacji</h2>", unsafe_allow_html=True)
    
    if st.session_state.prediction is not None and st.session_state.uploaded_image is not None:
        # Podziel prawy panel na dwie części
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            # Wyświetlenie maski segmentacji
            st.markdown("<h3>Maska segmentacji</h3>", unsafe_allow_html=True)
            colored_mask = colorize_mask(st.session_state.prediction)
            st.image(colored_mask, caption="Maska segmentacji", use_container_width=True)
            
        with result_col2:
            # Wyświetlenie nałożenia maski na oryginał
            st.markdown("<h3>Nałożenie na oryginał</h3>", unsafe_allow_html=True)
            overlay = overlay_masks(st.session_state.uploaded_image, st.session_state.prediction)
            st.image(overlay, caption="Nałożenie segmentacji na oryginał", use_container_width=True)
        
        # Wyświetlenie metryk
        st.markdown("<h3>Metryki modelu</h3>", unsafe_allow_html=True)
        
        metrics = st.session_state.metrics
        metrics_cols = st.columns(3)
        
        with metrics_cols[0]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Dice Score</h4>
                    <h2>{metrics.get('dice', 'N/A'):.4f}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with metrics_cols[1]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>IoU</h4>
                    <h2>{metrics.get('iou', 'N/A'):.4f}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with metrics_cols[2]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Accuracy</h4>
                    <h2>{metrics.get('accuracy', 'N/A'):.4f}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Szczegółowe metryki dla każdej klasy, jeśli są dostępne
        if 'class_metrics' in metrics:
            st.markdown("<h4>Metryki dla poszczególnych klas</h4>", unsafe_allow_html=True)
            
            class_names = ["Tło", "Klasa 1 (NCR/NET)", "Klasa 2 (ED)", "Klasa 3 (ET)"]
            class_metrics = metrics['class_metrics']
            
            for i, class_name in enumerate(class_names):
                if str(i) in class_metrics:
                    class_data = class_metrics[str(i)]
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h4>{class_name}</h4>
                            <p>Dice: {class_data.get('dice', 'N/A'):.4f} | 
                               IoU: {class_data.get('iou', 'N/A'):.4f} | 
                               Precision: {class_data.get('precision', 'N/A'):.4f} | 
                               Recall: {class_data.get('recall', 'N/A'):.4f}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
    else:
        st.info("Prześlij zdjęcie i kliknij 'Analizuj zdjęcie' aby zobaczyć wyniki segmentacji.")

# Dodatkowe informacje o modelu
with st.expander("Informacje o modelu"):
    st.markdown("""
    Ten model segmentacji MRI mózgu jest zoptymalizowany do identyfikacji następujących regionów:
    
    - **Klasa 1 (Czerwony)**: NCR/NET - obszar nekrotyczny i nie wzmacniający się guz
    - **Klasa 2 (Zielony)**: ED - obrzęk okołoguzowy
    - **Klasa 3 (Niebieski)**: ET - wzmacniający się guz
    
    Model został przeszkolony na zbiorze danych BraTS (Brain Tumor Segmentation) i wykorzystuje
    architekturę U-Net do wykonywania segmentacji.
    """)

# Instrukcje dla użytkownika jak skonfigurować Google Colab
with st.expander("Jak skonfigurować Google Colab"):
    st.markdown("""
    ### Konfiguracja Google Colab do działania z tą aplikacją:
    
    1. Otwórz nowy notatnik Google Colab.
    
    2. Zainstaluj potrzebne biblioteki:
    ```python
    !pip install torch torchvision flask pyngrok pillow numpy
    ```
    
    3. Stwórz prosty serwer Flask, który będzie udostępniał Twój model:
    ```python
    from flask import Flask, request, jsonify
    from pyngrok import ngrok
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    import io
    
    # Tutaj zaimportuj swój model
    # from model import YourModel
    
    # Inicjalizacja aplikacji Flask
    app = Flask(__name__)
    
    # Tutaj załaduj swój model
    # model = YourModel()
    # model.load_state_dict(torch.load('your_model.pth'))
    # model.eval()
    
    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Przetwarzanie obrazu
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        input_tensor = transform(image.convert('L')).unsqueeze(0)
        
        # Tutaj wykonaj predykcję swoim modelem
        # with torch.no_grad():
        #     prediction = model(input_tensor)
        #     mask = prediction.argmax(1).squeeze().cpu().numpy()
        
        # Dla celów demonstracyjnych, tworzymy przykładową maskę
        mask = np.random.randint(0, 4, size=(256, 256), dtype=np.uint8)
        
        # Przykładowe metryki
        metrics = {
            'dice': 0.85,
            'iou': 0.75,
            'accuracy': 0.90,
            'class_metrics': {
                '0': {'dice': 0.95, 'iou': 0.92, 'precision': 0.94, 'recall': 0.96},
                '1': {'dice': 0.80, 'iou': 0.70, 'precision': 0.82, 'recall': 0.78},
                '2': {'dice': 0.82, 'iou': 0.72, 'precision': 0.81, 'recall': 0.83},
                '3': {'dice': 0.83, 'iou': 0.73, 'precision': 0.84, 'recall': 0.82}
            }
        }
        
        return jsonify({
            'mask': mask.tolist(),
            'metrics': metrics
        })
    
    # Uruchomienie ngrok do udostępnienia aplikacji
    ngrok_tunnel = ngrok.connect(5000)
    print('Public URL:', ngrok_tunnel.public_url)
    
    # Uruchomienie serwera Flask
    if __name__ == '__main__':
        app.run(port=5000)
    ```
    
    4. W miejsce komentarzy dodaj kod ładujący Twój własny model i wykonujący predykcję.
    
    5. Po uruchomieniu kodu, ngrok wygeneruje publiczny URL. Skopiuj ten URL i wklej go w pole "URL do Google Colab" w aplikacji Streamlit.
    
    6. Pamiętaj, że darmowe sesje ngrok wygasają po pewnym czasie, więc w przypadku dłuższego używania warto rozważyć inne metody udostępniania modelu.
    """)

# Informacje o autorze
st.sidebar.markdown("### O aplikacji")
st.sidebar.info(
    "Ta aplikacja umożliwia segmentację obrazów MRI mózgu z wykorzystaniem modelu AI "
    "hostowanego na Google Colab. Aplikacja wykonuje komunikację z zewnętrznym modelem "
    "za pomocą API."
)

# Dodanie linku do GitHub
st.sidebar.markdown("[Kod źródłowy na GitHub](https://github.com/twoja-nazwa-uzytkownika/mri-segmentation)")