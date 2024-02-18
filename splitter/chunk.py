import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk(chunk_size):
    csv_doc=list()
    directory_path = "data"
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            loader = CSVLoader(file_path)
            csv_doc.extend(loader.load())
            
    pdf_doc=[]
    loader = PyPDFDirectoryLoader("data")
    pdf_doc = loader.load()
    docs=pdf_doc + csv_doc
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size , chunk_overlap =20)
    text_chunks= text_splitter.split_documents(docs)
    return text_chunks
    
