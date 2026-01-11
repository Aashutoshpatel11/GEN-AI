from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatHuggingFace( llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
) )

parser = StrOutputParser()

def wordCount(string):
    return len(string.split())

template = PromptTemplate(
    template="Write joke 1 line on {topic}",
    input_variables=['topic']
)

chain = template | model | parser | RunnableLambda(wordCount)

print( chain.invoke({'topic':"AI"}) )