from dotenv import load_dotenv
import os
load_dotenv()
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


# Load PDF and create FAISS DB (once)
def load_pdf_and_create_index(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

# Load both documents
litigation_index = load_pdf_and_create_index("data/Guide-to-Litigation-in-India.pdf")
icai_index = load_pdf_and_create_index("data/Legal-Compliance-and-Corporate-Laws-ICAI.pdf")

# Fetch relevant sections
def get_relevant_legal_text(query):
    results_lit = litigation_index.similarity_search(query, k=2)
    results_icai = icai_index.similarity_search(query, k=2)
    return [doc.page_content for doc in results_lit + results_icai]