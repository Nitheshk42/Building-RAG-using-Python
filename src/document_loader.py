from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    return documents