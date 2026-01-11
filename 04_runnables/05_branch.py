from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatHuggingFace( llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
) )

parser = StrOutputParser()

reportTemplate = PromptTemplate(
    template='Write a detailed report about {topic}',
    input_variables=['topic']
)

summaryTemplate = PromptTemplate(
    template='Write a short summary about report \n {topic}', 
    input_variables=['topic']
)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, summaryTemplate | model | parser),
    RunnablePassthrough()
)

chain = reportTemplate | model | parser | branch_chain

print( chain.invoke({'topic':"AI"}) )