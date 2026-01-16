from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('data/data.pdf')
docs = loader.load()

print(docs[0])