from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

template = load_prompt("template.json")

st.header("CUSTOM GPT")

paper_input = st.selectbox('Select Topic',["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox('Select Style', ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )
length_input = st.selectbox('Select Length', ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

if st.button('Summarize'):
    chain = template | model
    result = chain.stream({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    print(result)
    st.write(result)
