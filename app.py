import streamlit as st
import pandas as pd
import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import faiss
from diffusers import StableDiffusionPipeline


# Load dataset

CSV_PATH = "D:/SSN/SEMESTER 1/DS/MEDIVAL/newsimages/subset.csv"

df = pd.read_csv(CSV_PATH)
df.columns = ["id", "url", "headline", "entities", "hash", "image_url"]


# Load CLIP model
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_clip_model()


# Precompute image embeddings

@st.cache_data(show_spinner=False)
def compute_image_embeddings(df, _model, _preprocess, _device):
    embeddings = []
    for url in df["image_url"]:
        try:
            response = requests.get(url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_input = _preprocess(image).unsqueeze(0).to(_device)
            with torch.no_grad():
                emb = _model.encode_image(image_input)
            emb = emb.cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)  # normalize
            embeddings.append(emb)
        except Exception:
            # fallback in case of bad/missing image
            embeddings.append(np.zeros(_model.visual.output_dim))
    return np.array(embeddings)

image_embeddings = compute_image_embeddings(df, model, preprocess, device)


# Build FAISS index

@st.cache_resource
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cosine if normalized)
    index.add(embeddings.astype("float32"))
    return index

faiss_index = build_faiss_index(image_embeddings)

def retrieve_with_faiss(query_embedding, index, k=3):
    D, I = index.search(query_embedding[np.newaxis, :].astype("float32"), k)
    return I[0]


# Load Stable Diffusion generator

@st.cache_resource
def load_generator_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    return pipe

generator = load_generator_model()


# Streamlit App

st.set_page_config(page_title="MediaEval NewsImages", layout="wide")
st.title("📰 MediaEval NewsImages Demo (CLIP + FAISS + Stable Diffusion)")

# Initialize session state for persistence
if "retrieved" not in st.session_state:
    st.session_state["retrieved"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = None

# Input
headline_input = st.text_area("Enter headline or description:")

# Retrieval
if st.button("Retrieve Images"):
    if not headline_input.strip():
        st.warning("⚠️ Please enter a headline first.")
    else:
        with torch.no_grad():
            text_tokens = clip.tokenize([headline_input]).to(device)
            text_embedding = model.encode_text(text_tokens).cpu().numpy()[0]
            text_embedding = text_embedding / np.linalg.norm(text_embedding)

        topk_idx = retrieve_with_faiss(text_embedding, faiss_index, k=3)
        st.session_state["retrieved"] = df.iloc[topk_idx].to_dict("records")

# Generation
if st.button("Generate New Image"):
    if not headline_input.strip():
        st.warning("⚠️ Please enter a headline first.")
    else:
        with st.spinner("🎨 Generating image..."):
            gen_img = generator(headline_input).images[0]
            st.session_state["generated"] = gen_img

# -------------------------------
# Display Results
# -------------------------------
if st.session_state["retrieved"]:
    st.subheader("Retrieved Images (CLIP + FAISS)")
    cols = st.columns(len(st.session_state["retrieved"]))
    for i, row in enumerate(st.session_state["retrieved"]):
        with cols[i]:
            st.image(row["image_url"], caption=f"ID: {row['id']} | {row['headline']}", use_container_width=True)

if st.session_state["generated"] is not None:
    st.subheader("Generated Image (Stable Diffusion)")
    st.image(st.session_state["generated"], caption=f"Generated for: {headline_input}", use_container_width=True)
