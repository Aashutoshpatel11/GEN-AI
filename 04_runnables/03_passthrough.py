from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatHuggingFace( llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
) )

topicNamesTemplate = PromptTemplate(
    template='give name of 5 topics of {topic} without any definitons, just give names',
    input_variables=['topic']
)

definitionTemplate = PromptTemplate(
    template='Write small definitions of all 5 topics\n {topicNames}',
    input_variables=['topicNames']
)

parser = StrOutputParser()

first_chain = topicNamesTemplate | model | parser

second_chain = RunnableParallel({
    '5 topics': RunnablePassthrough() ,
    'Definitions': definitionTemplate | model | parser
})

final_chain = first_chain | second_chain

print( final_chain.invoke({'topic':'AI'}) )