from langchain_community.retrievers import WikipediaRetriever


retreiver = WikipediaRetriever(
    
    top_k_results=2,
    lang='en'
)

result = retreiver.invoke('AI')

print(result)