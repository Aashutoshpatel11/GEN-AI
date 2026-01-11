from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

system_msg = 'You are a human assistant'

messages = [
    system_msg
]

while True:
    userInput = input("\nYOU:")
    if( userInput == 'exit' ):
        break

    messages.append(HumanMessage( content= userInput ))

    response = model.invoke(messages)

    messages.append(AIMessage( content=response.content ))

    print('\nAI:', response.content)
