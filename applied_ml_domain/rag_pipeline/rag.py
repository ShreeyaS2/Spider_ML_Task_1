import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

PAPERS_DIR = r"C:\Personal\comp_proj\spider_ml_task1\applied_ml_domain\rag_pipeline\papers"
CHROMA_DIR = r"C:\Personal\comp_proj\spider_ml_task1\applied_ml_domain\rag_pipeline\chroma_db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5" 

load_dotenv()  

#Ingesting papers and splitting into chunks
def ingest_papers():
    with st.spinner("Ingesting papers..."):
        pdf_files = DirectoryLoader(PAPERS_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        files = pdf_files.load()
    if not files:
        st.write(f"No PDFs found")
        return []

    papers = set()
    for file in files:
        name= os.path.basename(file.metadata.get("source", ""))
        file.metadata["paper"] = name
        papers.add(name)

    st.write(f"Found {len(papers)} papers, {len(files)} pages. Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=70)
    chunks = splitter.split_documents(files) #splits a list of documents into smaller chunks based on the specified chunk size and overlap. Each chunk is treated as a separate document, allowing for more manageable processing in subsequent steps.
    st.write(f"Total chunks created: {len(chunks)}")
    return chunks

#Embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})

#Building and loading vectorstore
def build_vectorstore():
    chunks = ingest_papers()
    if not chunks:
        return None
    with st.spinner("Building vectorstore..."):
        embeddings = get_embeddings()
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR, collection_name="papers")
    st.write("Vectorstore ready.\n")
    return vectorstore

def load_vectorstore():
    with st.spinner("Loading vectorstore..."):
        embeddings = get_embeddings()
        return Chroma(collection_name="papers", persist_directory=CHROMA_DIR, embedding_function=embeddings)

#Retrieve context
def retrieve(vectorstore, query):
    with st.spinner("Retrieving relevant chunks..."):
        return vectorstore.max_marginal_relevance_search(query, k=6)

#Ranking the context on the basis of relevance
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def rerank(query, docs):
    pairs = []
    for doc in docs:
        pair = (query, doc.page_content)
        pairs.append(pair)
    scores = reranker.predict(pairs)
    ranked_scores = sorted(zip(scores, docs), reverse=True)
    results=[]
    for _, doc in ranked_scores[:3]:
        results.append(doc)
    return results

#Generate answer
def generate_answer(query, docs):
    with st.spinner("Generating answer..."):
        context_parts = []
        for doc in docs:
            paper = doc.metadata.get("paper", "unknown")
            context_parts.append(f"[Source: {paper}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        prompt = f"""You are a research assistant answering questions about NLP and LLM research papers. Use only the context below to answer in a clear, precise and educated manner. Mention the source paper(s) in your answer. If the context lacks sufficient information, say so directly and do not make up any information.

    Context:
    {context}

    Question: {query}

    Answer:"""
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",max_tokens=4000, api_key=os.getenv("GEMINI_API_KEY"))
        response = llm.invoke(prompt)
        return response.content

def run():
    st.title("RAG Pipeline Application")
    st.write("This application allows you to go through a collection of NLP research papers using a Retrieval-Augmented Generation (RAG) pipeline. It ingests PDF papers, splits them into chunks, creates embeddings, and retrieves relevant information to answer your queries.")
    os.makedirs(PAPERS_DIR, exist_ok=True)
    if "vectorstore" not in st.session_state:
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            st.session_state.vectorstore = load_vectorstore()
        else:
            st.session_state.vectorstore = build_vectorstore()
            if st.session_state.vectorstore is None:
                st.write(f"Please add PDF papers to the folder and run again.")
                return

    st.write("RAG Pipeline Application ready. Type 'quit' to exit.\n")

    query = st.text_input("Query: ").strip()
    
    if query.lower() == "quit":
        st.write("Exiting RAG Pipeline. Goodbye!")
        return
    elif query:
        docs = retrieve(st.session_state.vectorstore, query)
        docs = rerank(query, docs)
        answer = generate_answer(query, docs)

        st.divider()
        st.header("Answer:")
        st.write(answer)
        with st.expander("Show retrieved chunks"):
            with st.spinner("Loading retrieved chunks..."):
                st.header("\nRetrieved Chunks:")
                for i, doc in enumerate(docs):
                    paper = doc.metadata.get("paper", "unknown")
                    page = doc.metadata.get("page", "unknown")
                    excerpt = doc.page_content[:400].replace("\n", " ").strip()
                    st.subheader(f"{i+1}. Paper: {paper}, Page: {page}")
                    st.write(f"{excerpt}...")
                    st.divider()



if __name__ == "__main__":
    run()