import os
import asyncio
import time

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = ''

custom_prompt_template = """
Question: {question}
Answer: Translate the Question to Vietnamese.
"""

def llm():
    llm = OpenAI(temperature=0.9)
    return llm

def translator():
    prompt = PromptTemplate(
        input_variables=["question"],
        template=custom_prompt_template,
    )
    llm = llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain




