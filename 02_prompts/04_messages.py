from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# system_msg = SystemMessage('You are music expert')
# human_msg = HumanMessage('how to learn piano in one day')
# message = [system_msg, human_msg]

message = [
    SystemMessage('You are music expert'),
    HumanMessage('how to learn piano in one day')
]

result = model.invoke(message)

message.append( AIMessage(content=result.content) )

print(message)