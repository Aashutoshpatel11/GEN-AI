from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

template = ChatPromptTemplate({
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
})

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# prompt = template.invoke({'domain':'IT Professional', 'topic':'AI'})
# print(prompt)

chain = template | model

print( chain.invoke({ 'domain': "AI", 'topic': 'AI' }) )