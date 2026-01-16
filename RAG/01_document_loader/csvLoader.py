from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('data/csv.csv')

docs = loader.load()

print(docs)