import re, cv2, numpy as np, math, collections
from PIL import Image
from unidecode import unidecode
from rapidfuzz import fuzz

# Stopwords comuns em embalagem
STOPWORDS = set("""
COMPRIMIDO COMPRIMIDOS CAPSULA CAPSULAS VIA ORAL USO ORAL ADULTO INFANTIL GOTAS SUSPENSAO
MG ML % SOLUCAO REVESTIDO REVESTIDA NOVA FORMULA CONTEM CAIXA CARTELA BULA
MEDICAMENTO GENERICO GENERICO EUROFARMA MEDLEY EMS ACHE BIOSINTETICA TEUTO NEOQUIMICA
""".split())

def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unidecode(s).upper()
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    s = normalize_text(s)
    toks = [t for t in re.split(r"[^A-Z0-9]+", s) if t and len(t) >= 3 and t not in STOPWORDS]
    return toks

def extract_mg(s: str):
    s = normalize_text(s)
    vals = re.findall(r"\b(\d{1,4})\s*MG\b", s)
    return sorted(set(int(v) for v in vals))

# ---------------- OCR ----------------
def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def make_variants(img: Image.Image):
    """Pré-processamento simplificado para performance"""
    cv = _pil_to_cv(img)
    out = []
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    th = cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)
    out.append(_cv_to_pil(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)))
    return out

def _tesseract(img: Image.Image, config="", lang="por+eng"):
    try:
        import pytesseract, numpy as np
        arr = np.array(img.convert("L"))
        return pytesseract.image_to_string(arr, lang=lang, config=config)
    except Exception:
        return ""

def _easyocr(img: Image.Image):
    try:
        import easyocr, numpy as np
        reader = easyocr.Reader(["pt","en"], gpu=False)
        res = reader.readtext(np.array(img.convert("RGB")), detail=0, paragraph=True)
        return "\n".join(res)
    except Exception:
        return ""

def run_ocr_text(img: Image.Image, user_words_path=None, use_easyocr=True):
    variants = [img] + make_variants(img)
    texts = []
    # Primeiro tenta com Tesseract
    for v in variants:
        for psm in (6,7):  # apenas configs mais relevantes
            cfg = f"--oem 3 --psm {psm}"
            if user_words_path:
                cfg += f" --user-words {user_words_path}"
            t = _tesseract(v, config=cfg)
            if t and len(t.strip())>0: 
                texts.append(t)
    # Só se falhar, tenta EasyOCR
    if use_easyocr and not texts:
        texts.append(_easyocr(img))
    best = max(texts, key=lambda s: len(s), default="")
    return best, variants[1] if len(variants)>1 else img

# ---------------- Matching ----------------
def build_idf(choices_norm):
    N = len(choices_norm)
    df = collections.Counter()
    for s in choices_norm:
        df.update(set(tokenize(s)))
    idf = {}
    for tok, c in df.items():
        idf[tok] = math.log((N + 1) / (c + 1)) + 1.0
    return idf

def weighted_jaccard(tokens_a, tokens_b, idf):
    A, B = set(tokens_a), set(tokens_b)
    inter = A & B
    union = A | B
    if not union: return 0.0
    w_inter = sum(idf.get(t,1.0) for t in inter)
    w_union = sum(idf.get(t,1.0) for t in union)
    return w_inter / w_union

def smart_best_match(ocr_text, choices_raw, choices_norm, idf):
    ocr_toks = tokenize(ocr_text)
    mg_vals = extract_mg(ocr_text)
    best = None
    best_score = 0.0
    for i, s_norm in enumerate(choices_norm):
        s_raw = choices_raw[i]
        s_toks = tokenize(s_norm)
        j = weighted_jaccard(ocr_toks, s_toks, idf)
        f = fuzz.WRatio(" ".join(ocr_toks), " ".join(s_toks)) / 100.0
        score = 0.65*j + 0.35*f
        prod_mg = extract_mg(s_raw)
        if mg_vals and prod_mg and set(mg_vals) & set(prod_mg):
            score += 0.10
        if score > best_score:
            best_score = score
            best = i
    return best, best_score
