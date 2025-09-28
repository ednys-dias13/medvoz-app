import os, io, pandas as pd, streamlit as st
from PIL import Image
from gtts import gTTS
from utils import normalize_text, run_ocr_text, build_idf, smart_best_match

st.set_page_config(page_title="MedVoz • OCR Texto v3.0", page_icon="💊", layout="centered")
st.title("💊 MedVoz — OCR Texto Automático (v3.0)")
st.caption("Agora com captura automática + áudio sempre ativo para maior acessibilidade.")

# Sidebar
st.sidebar.header("Configurações")
# Remove caminho local e usa arquivo direto
base_path = "lista_medicamentos_pfpb_ean_marco_2025.xlsx"
use_easyocr = st.sidebar.checkbox("Usar EasyOCR como fallback", True)
show_debug = st.sidebar.checkbox("Mostrar depuração", False)

@st.cache_data
def load_base(xlsx_path: str):
    try:
        # Tenta carregar do diretório atual (Streamlit Cloud)
        df = pd.read_excel(xlsx_path, sheet_name=0)
    except:
        # Fallback: cria um dataframe de exemplo se o arquivo não existir
        st.warning("⚠️ Arquivo base não encontrado. Usando dados de exemplo.")
        df = pd.DataFrame({
            "PRODUTO": ["Paracetamol 500mg", "Dipirona 500mg", "Ibuprofeno 400mg"],
            "INDICAÇÃO": ["Analgésico e antitérmico", "Analgésico e antitérmico", "Anti-inflamatório"]
        })
    
    df["__PRODUTO_NORM__"] = df["PRODUTO"].astype(str).map(normalize_text)
    df["__INDICACAO_NORM__"] = df["INDICAÇÃO"].astype(str).map(normalize_text)
    idf = build_idf(df["__PRODUTO_NORM__"].tolist())
    return df, idf

with st.spinner("Carregando base..."):
    df, idf = load_base(base_path)
    st.success(f"Base carregada: {len(df):,} registros.".replace(",", "."))

# Função para falar sempre
def speak(text: str):
    try:
        tts = gTTS(text=text, lang="pt", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        st.audio(buf, format="audio/mp3", autoplay=True)
    except Exception:
        st.info("⚠️ Não consegui sintetizar voz agora.")

def answer(row):
    return f"Este é **{row['PRODUTO']}**. Indicação principal: **{row['INDICAÇÃO']}**."

# Captura automática
img = st.camera_input("📸 Centralize a frente da caixa do medicamento")
if img is not None:
    uploaded_image = Image.open(img)
    with st.spinner("🔎 Lendo texto da embalagem..."):
        # Ajusta o caminho do user_words
        user_words_path = "user_words.txt" if os.path.exists("user_words.txt") else None
        ocr_text, preview_img = run_ocr_text(uploaded_image, user_words_path=user_words_path, use_easyocr=use_easyocr)
        best_idx, best_score = smart_best_match(ocr_text, df["PRODUTO"].tolist(), df["__PRODUTO_NORM__"].tolist(), idf)

        if best_idx is not None and best_score >= 0.10:
            row = df.iloc[best_idx]
            st.success("✅ Remédio identificado!")
            st.markdown(answer(row))
            speak(f"Este é {row['PRODUTO']}. Indicação principal: {row['INDICAÇÃO']}.")
        else:
            st.error("⚠️ Confiança baixa para identificar pelo texto.")
            speak("Não consegui identificar. Por favor, aproxime mais a caixa do medicamento.")

        if show_debug:
            st.divider()
            st.subheader("Depuração")
            st.caption("Texto OCR")
            st.code(ocr_text or "(vazio)")
            st.caption("Score")
            st.write(best_score)
            st.caption("Pré-processada (uma variante)")
            st.image(preview_img, use_column_width=True)

st.divider()
st.markdown("ℹ️ Este protótipo identifica **automaticamente** pelo texto da embalagem (sem EAN).")   