import os
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

class TextFileLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        print("Reading Text file: ", self.file_path)
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return [Document(page_content=text, metadata={'source': self.file_path})]

def get_content_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def create_vector_db(data_path, faiss_db_path):
    print("---------------------------------------------------------------")
    raw_documents = []

    # Load PDFs
    pdf_loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()
    [print("Retrieved PDF file: ", doc.metadata['source']) for doc in pdf_documents]
    raw_documents.extend(pdf_documents)
    print(len(pdf_documents), "PDFs loaded.")

    # Load .txt files
    txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
    for txt_file in txt_files:
        txt_loader = TextFileLoader(os.path.join(data_path, txt_file))
        txt_documents = txt_loader.load()
        raw_documents.extend(txt_documents)

    # Remove duplicate documents based on content hash
    seen_hashes = set()
    unique_documents = []

    for doc in raw_documents:
        content_hash = get_content_hash(doc.page_content)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        unique_documents.append(doc)

    print(f"{len(unique_documents)} unique documents retained after deduplication.")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(unique_documents)
    print(len(texts), "chunks created.")

    # Embed and store
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large',
                                       model_kwargs={'device': 'cuda'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(faiss_db_path)

    print("Vector DB created and saved at:", faiss_db_path)
    print("---------------------------------------------------------------")


