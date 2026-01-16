from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

loader = DirectoryLoader(path='data', glob='*.pdf', loader_cls=PyPDFLoader)

docs = loader.load()

print (docs)