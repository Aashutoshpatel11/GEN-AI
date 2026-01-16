from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings( model='gemini-embedding-001' )

vector_store = Chroma(
    embedding_function=embeddings,
    collection_name='sample',
    persist_directory='mydb'
)

retreiver = vector_store.as_retriever(
    search_kwargs={"k": 2},
    search_type='mmr'
    )

result = retreiver.invoke("What is Chroma used for?")

print(result)