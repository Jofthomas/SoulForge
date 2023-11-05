from llama_index.llms import ChatMessage
from llama_index.tools import BaseTool
from llama_index.llms.base import LLM
from abc import ABC, abstractmethod
from config import *
from typing import (
    Sequence,
    List,
    Optional
)
from ..personality import Personality

import nest_asyncio
nest_asyncio.apply()

class BaseNPC(ABC):
    """LLM-agnostic base NPC class template

    # TODO add validators
    """
    llm: LLM
    actions: Sequence[BaseTool] = [],
    chat_history: List[ChatMessage] = [],
    personality: Optional[str | Personality] = None,

    def reset(self) -> None:
        if self._personality is not None:
            self._chat_history = [
                ChatMessage(content=self._personality.get_description(), role="system")
            ]
        else:
            self._chat_history = []

    @abstractmethod
    def chat(
            self, 
             message: str
        ) -> str:
        """
        Run LLM chat completion: 
        - update chat history
        - access memory
        - send message to LLM (optional tool usage: _call_function method)
        - return text content of the LLM response
        """

    @abstractmethod
    def _call_function(
            self, 
            function_call: dict
        ) -> ChatMessage:
        """
        Use tool:
        - invoke by function_call dict key
        - return the ChatMessage object with tool output as text content
        """

    @abstractmethod
    def live_chat(self) -> None:
        """
        Interactive chat completion between the user and LLM.
        Uses chat interface with optional clause to end the conversation.
        LLM tools output may call this clause as well. 
        """
        print("You are now chatting with the NPC. Type 'exit()' to end the conversation.")
        while True:
            # Get player input
            player_input = input("You: ")
            if player_input.lower() == 'exit()':
                print("You ended the conversation.")
                break
            
            # Live chat implementation
