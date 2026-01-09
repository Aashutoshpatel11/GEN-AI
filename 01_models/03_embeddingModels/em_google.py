from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')

vector = embeddings.embed_documents(documents) # for list/array of data
vector2 = embeddings.embed_query(documents) # for single data

print(vector)