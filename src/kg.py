
import logging
import sys
import time

from llama_index import (
    KnowledgeGraphIndex,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.llms import OpenAI
from llama_index.graph_stores import SimpleGraphStore
from llama_index.storage.storage_context import StorageContext
from llama_index.schema import Document
import openai

from config import *

openai.api_key = OPENAI_KEY

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class Knowlege_Graph:
    """Knowlege_Graph class for keeping conversation as knowledge graph format
    """
    def __init__(
        self, 
        llm: OpenAI = OpenAI(temperature=0, model="text-davinci-002"),
        graph_store = SimpleGraphStore(),
        documents = SimpleDirectoryReader("../data_tmp").load_data()
    ):
        self.llm = llm
        self.graph_store = graph_store
        self.documents = documents
        self.service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)
        self.storage_context = StorageContext.from_defaults(graph_store=graph_store)

        self.index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            max_triplets_per_chunk=2,
            service_context=self.service_context,
            include_embeddings=True,
        )

        self.query_engine = self.index.as_query_engine(
            include_text=False, response_mode="tree_summarize"
        )


    def query(self, user_question):
        response = self.query_engine.query(user_question)
        #if response=="There is not enough information to answer the query.":
        return response


    def save_conversation(self, chat_history):
        new_data = [Document(text=chat_history)]
        self.index._insert(new_data)



if __name__ == "__main__":
    kg = Knowlege_Graph()
    time.sleep(61)
    res = kg.query("What is K'thrax")
    print(res)
    time.sleep(61)
    res = kg.query("Tell me about America")
    print(res)
    kg.save_conversation("America is a country located in North America.")
    time.sleep(61)
    res = kg.query("Tell me about America")
    print(res)