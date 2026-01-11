from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatHuggingFace( llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
) )

template1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Write a fact about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

runnable = RunnableParallel({
    'joke': template1 | model | parser,
    'fact': template2 | model | parser,
})

print ( runnable.invoke({'topic':'AI'}) )
