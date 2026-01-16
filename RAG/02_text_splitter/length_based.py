from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader('data/data.pdf')
data = loader.load()

print(type(data))

splitter = CharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=0
)

result = splitter.split_documents(data)

print(len(result), result[0])

