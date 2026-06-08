import os
import json
import fitz
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv, find_dotenv
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy
import faiss

PAPERS_DIR     = r"C:\Personal\comp_proj\spider_ml_task1\applied_ml_domain\rag_pipeline\papers"
IMAGES_DIR     = r"C:\Personal\comp_proj\spider_ml_task1\applied_ml_domain\rag_pipeline\images"
FAISS_TEXT_DIR = r"C:\Personal\comp_proj\spider_ml_task1\applied_ml_domain\rag_pipeline\faiss_text_db"
FAISS_IMAGE_DIR= r"C:\Personal\comp_proj\spider_ml_task1\applied_ml_domain\rag_pipeline\faiss_image_db"
EMBED_MODEL    = "BAAI/bge-small-en-v1.5"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(find_dotenv())
load_dotenv(find_dotenv(), override=True)

print("API KEY:", os.getenv("GEMINI_API_KEY"))

#Ingesting papers and chunking
def ingest_papers():
    with st.spinner("Ingesting papers..."):
        pdf_files = DirectoryLoader(PAPERS_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        files = pdf_files.load()
    if not files:
        st.write("No PDFs found")
        return []

    papers = set()
    for file in files:
        name = os.path.basename(file.metadata.get("source", ""))
        file.metadata["paper"] = name
        papers.add(name)

    st.write(f"Found {len(papers)} papers, {len(files)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=70)
    chunks = splitter.split_documents(files)
    st.write(f"Total chunks created: {len(chunks)}")
    return chunks

#Extracting images from papers
def extract_images():
    with st.spinner("Extracting images..."):
        pdf_files = DirectoryLoader(PAPERS_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        files = pdf_files.load()
    if not files:
        st.write("No PDFs found")
        return []

    image_metadata = []
    for file in files:
        doc = fitz.open(file.metadata.get("source", ""))
        paper = os.path.basename(file.metadata.get("source", ""))
        for page_num in range(len(doc)):
            page = doc[page_num]
            for i, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                filepath = os.path.join(IMAGES_DIR, f"image_{page_num}_{i}.{ext}")
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                image_metadata.append({"path": filepath, "paper": paper, "page": page_num + 1})
    return image_metadata

#Embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )


@st.cache_resource
def get_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

#Image embeddings
def image_embeddings(image_metadata):
    model, processor = get_clip_model()
    embeddings = []
    valid_metadata = []
    for i, item in enumerate(image_metadata):
        print(f"Processing image {i+1}/{len(image_metadata)}", end="\r")
        try:
            image = Image.open(item["path"]).convert("RGB")
            if image.width < 100 or image.height < 100:
                continue
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)
            embeddings.append(embedding[0].squeeze().cpu().tolist())
            valid_metadata.append(item)
        except Exception as e:
            print(f"Skipping {item['path']}: {e}")
    return embeddings, valid_metadata

#Query embedding for images
def query_clip(query):
    model, processor = get_clip_model()
    inputs = processor(text=query, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs)
    return query_embedding[0].cpu().numpy()

#Build and load vectorstore
def build_vectorstore():
    chunks = ingest_papers()
    if not chunks:
        return None
    with st.spinner("Building text vectorstore..."):
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_TEXT_DIR)
    st.write("Text vectorstore ready.")
    return vectorstore


def load_vectorstore():
    with st.spinner("Loading text vectorstore..."):
        if not os.path.exists(FAISS_TEXT_DIR):
            raise ValueError("FAISS text index does not exist yet.")
        embeddings = get_embeddings()
        return FAISS.load_local(FAISS_TEXT_DIR, embeddings)

#Build and load image vectorstore
def build_image_vectorstore(image_metadata):
    embeddings, valid_metadata = image_embeddings(image_metadata)
    if not embeddings:
        return None, []

    with st.spinner("Building image vectorstore..."):
        os.makedirs(FAISS_IMAGE_DIR, exist_ok=True)
        vectors = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(vectors.shape[1]) 
        index.add(vectors)

        faiss.write_index(index, os.path.join(FAISS_IMAGE_DIR, "images.index"))
        with open(os.path.join(FAISS_IMAGE_DIR, "metadata.json"), "w") as f:
            json.dump(valid_metadata, f)

    st.write("Image vectorstore ready.")
    return index, valid_metadata


def load_image_vectorstore():
    with st.spinner("Loading image vectorstore..."):
        index_path = os.path.join(FAISS_IMAGE_DIR, "images.index")
        meta_path  = os.path.join(FAISS_IMAGE_DIR, "metadata.json")
        if not os.path.exists(index_path):
            raise ValueError("FAISS image index does not exist yet.")
        index = faiss.read_index(index_path)
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        return index, metadata


#Retrieve image and context
def retrieve_images(image_store, query):
    with st.spinner("Retrieving relevant images..."):
        index, metadata = image_store
        query_vec = query_clip(query).astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vec)   
        _, indices = index.search(query_vec, k=1)
        top = indices[0][0]
        if top != -1 and top < len(metadata):
            return metadata[top]
        return None


def retrieve(vectorstore, query):
    with st.spinner("Retrieving relevant chunks..."):
        return vectorstore.max_marginal_relevance_search(query, k=6)


#Crossencoder for reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

def rerank(query, files):
    pairs = [(query, f.page_content) for f in files]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, files), reverse=True)
    return [files for _, files in ranked[:3]]

#Generate answer
def generate_answer(query, files):
    with st.spinner("Generating answer..."):
        context_parts = []
        for file in files:
            paper = file.metadata.get("paper", "unknown")
            context_parts.append(f"[Source: {paper}]\n{file.page_content}")
        context = "\n\n".join(context_parts)

        prompt = f"""You are a research assistant answering questions about NLP and LLM research papers. Use only the context below to answer in a clear, precise and educated manner. Mention the source paper(s) in your answer. If the context lacks sufficient information, say so directly and do not make up any information.

    Context:
    {context}

    Question: {query}

    Answer:"""

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=4000)
        )
        return response.text.strip()


def run():
    st.title("RAG Pipeline Application")
    st.write("This application allows you to go through a collection of NLP research papers using a Retrieval-Augmented Generation (RAG) pipeline. It ingests PDF papers, splits them into chunks, creates embeddings, and retrieves relevant information to answer your queries.")
    os.makedirs(PAPERS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    if "vectorstore" not in st.session_state:
        try:
            st.session_state.vectorstore = load_vectorstore()
        except Exception:
            st.session_state.vectorstore = build_vectorstore()
            if st.session_state.vectorstore is None:
                st.write("Please add PDF papers to the folder and run again.")
                return

    if "image_vectorstore" not in st.session_state:
        try:
            st.session_state.image_vectorstore = load_image_vectorstore()
        except Exception as e:
            print(f"Load failed: {e}")
            image_metadata = []
            for f in os.listdir(IMAGES_DIR):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                    image_metadata.append({"path": os.path.join(IMAGES_DIR, f)})
            print(f"Total images: {len(image_metadata)}")
            result = build_image_vectorstore(image_metadata)
            if result[0] is None:
                st.write("Please add PDF papers to the folder and run again.")
                return
            st.session_state.image_vectorstore = result

    st.write("RAG Pipeline Application ready. Type 'quit' to exit.")

    query = st.text_input("Query: ").strip()

    if query.lower() == "quit":
        st.write("Exiting RAG Pipeline. Goodbye!")
        return
    elif query:
        files = retrieve(st.session_state.vectorstore, query)
        files = rerank(query, files)
        answer = generate_answer(query, files)

        st.divider()
        st.header("Answer:")
        st.write(answer)

        images = retrieve_images(st.session_state.image_vectorstore, query)
        if images:
            st.divider()
            st.header("Most Relevant Image:")
            image = Image.open(images["path"])
            st.image(image)

        with st.expander("Show retrieved chunks"):
            with st.spinner("Loading retrieved chunks..."):
                st.header("Retrieved Chunks:")
                for i, file in enumerate(files):
                    paper = file.metadata.get("paper", "unknown")
                    page  = file.metadata.get("page", "unknown")
                    excerpt = file.page_content[:400].replace("\n", " ").strip()
                    st.subheader(f"{i+1}. Paper: {paper}, Page: {page}")
                    st.write(f"{excerpt}...")
                    st.divider()


if __name__ == "__main__":
    run()