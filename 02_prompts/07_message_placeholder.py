from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

load_dotenv()

template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())


prompt = template.invoke({'chat_history':chat_history,'query':"AI"})

print(prompt)