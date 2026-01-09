from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

document_vector = embedding.embed_documents(documents)
query_vector = embedding.embed_query("Kolkata")

scores = cosine_similarity([query_vector], document_vector)[0]

print(scores)