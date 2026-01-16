from langchain_community.document_loaders import TextLoader

loader = TextLoader('data/cricket.txt', encoding='utf-8')

data = loader.load()


print (data[0])