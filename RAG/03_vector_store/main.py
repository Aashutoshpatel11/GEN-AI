from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv  import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1, doc2, doc3, doc4, doc5]

load_dotenv()

vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001'),
    collection_name='sample',
    persist_directory='mydb'
)

vector_store.add_documents(docs)

# docsResult = vector_store.get()

# print(docsResult)

updatedDoc1 = Document(
    page_content="updated Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document('e6a9130d-99d9-4b12-ad35-420bc45b0e12', updatedDoc1)

# docsResult = vector_store.get()

# print(docsResult)


similarity = vector_store.similarity_search_with_score(query='virat', k=2)
# print(similarity)

vector_store.delete(ids=['c13b0e8a-8af3-47fe-95a9-e93cc24d43a0'])

docsResult = vector_store.get()

print(docsResult)