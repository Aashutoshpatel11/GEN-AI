from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

prompt = PromptTemplate(
    template='Write 5 facts on topic \n {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

chain = prompt | model

print( chain.invoke('India').content )