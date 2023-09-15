import asyncio
import time
import qa_bot as bot
import translator 

async def chat_bot(query):
    res = await bot.run(query)
    translator.run(res)