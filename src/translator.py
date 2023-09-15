import os
import asyncio
import time

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = ''

custom_prompt_template = """
You are a virtual assitant can translate any text input to {language}. You can auto detect the language of input text. You can return the language of input text belong the traslated text. The format is: [translated text]
"""

def llm():
    llm = OpenAI(temperature=0.9)
    return llm

def translator():
    prompt = PromptTemplate(
        input_variables=["language"],
        template=custom_prompt_template,
    )
    llm = llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def run(query):
    chain = translator()
    return chain.run(query)


