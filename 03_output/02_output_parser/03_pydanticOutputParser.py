from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatHuggingFace( llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
) )

class Person(BaseModel):
    name: str = Field(description='name of a Person')
    age: int = Field(description='age of a person')

parser = PydanticOutputParser( pydantic_object = Person )

template = PromptTemplate(
    template='Generate a random person from {country} \n {output_instructions}',
    input_variables=['country'],
    partial_variables={'output_instructions': parser.get_format_instructions() }
)

chain = template | model | parser

print( chain.invoke({'country':'India'}) )