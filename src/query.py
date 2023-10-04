import os
import torch
from transformers import pipeline
from typing import Optional, List, Mapping, Any

from llama_index import (
    ServiceContext, 
    SimpleDirectoryReader,
    SummaryIndex
)
from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CustomLLM, 
    CompletionResponse, 
    CompletionResponseGen,
    LLMMetadata,
)
from langchain.chains.conversation.memory import ConversationBufferMemory
from llama_index.llms.base import llm_completion_callback
from llama_index.agent import ReActAgent
from llama_index.tools import QueryEngineTool, ToolMetadata

from langchain.agents import Tool
from langchain.agents import initialize_agent
from text_generation import Client

from client import get_tgi_client
from config import MODEL_NAME

# set context window size
context_window = 4096
# set number of output tokens
num_output = 128

tgi_client = get_tgi_client()

# Set prompt params
sys_prompt = "A chat"
prompt = "What's my profession?"
prefix = "<|im_start|>"
suffix = "<|im_end|>\n"
sys_format = prefix + "system\n" + sys_prompt + suffix
user_format = prefix + "user\n" + prompt + suffix
assistant_format = prefix + "assistant\n"
input_text = sys_format + user_format + assistant_format

class LlongOrcaLLM(CustomLLM):

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=context_window,
            num_output=num_output,
            model_name=MODEL_NAME
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = tgi_client.generate(prompt=prompt, max_new_tokens=num_output, do_sample=True, temperature=1.0).generated_text

        return CompletionResponse(text=response)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()
    
# define our LLM
llm = LlongOrcaLLM()

service_context = ServiceContext.from_defaults(
    llm=llm, 
    embed_model="local:BAAI/bge-base-en-v1.5",
    context_window=context_window,
    num_output=num_output
)

# Load the your data
documents = SimpleDirectoryReader('./data').load_data()
index = SummaryIndex.from_documents(documents, service_context=service_context)
tools = [
    Tool(
        name="LlamaIndex",
        func=lambda q: str(index.as_query_engine().query(q)),
        description="The input to this tool should be a complete English sentence.",
        return_direct=True,
    ),
]

# Query and print response
query_engine = index.as_query_engine()
response = query_engine.query(input_text)

while True:
    text_input = input("User: ")
    response = query_engine.query(text_input)
    print(f'Agent: {response}')