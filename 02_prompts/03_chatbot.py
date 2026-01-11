from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

template = PromptTemplate(
    template='you are a assisstent, talk like a human and answer \n {text} ',
    input_variables=['text']
)

chain = template | model

inputText = ""

while( inputText != 'exit' ):
    inputText = input('YOU:') 
    result = chain.invoke(inputText)
    print('AI:',result.content)