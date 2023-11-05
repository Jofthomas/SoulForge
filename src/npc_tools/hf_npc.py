"""NPC implementation with HF models
"""
import json
from llama_index.llms.base import LLM
from llama_index.llms import (
    LangChainLLM, 
    ChatMessage,
)
from llama_index.callbacks import CallbackManager
from llama_index.tools import BaseTool
from langchain.base_language import BaseLanguageModel
from typing import (
    Sequence,
    List,
    Optional,
)

from .base import BaseNPC
from ..personality import Personality

class HuggingFaceNPC(BaseNPC):

    def __init__(
            self, 
            llm: LLM,
            actions: Sequence[BaseTool] = [],
            chat_history: List[ChatMessage] = [],
            personality: Optional[str | Personality] = None,
        ) -> None:
        
        self._llm = llm

        if len(actions) > 0:
            raise NotImplementedError("Tool usage is yet to be implemented.")
        
        # Add the personality message to chat history if personality is provided
        if personality is not None:
            chat_history.insert(0, ChatMessage(content=personality.get_description(), role="system"))
        self._chat_history = chat_history
        self._personality = personality

    @classmethod
    def from_langchain_llm(
        cls,
        llm: BaseLanguageModel, 
        callback_manager: Optional[CallbackManager] = None,
        **params,
    ) -> "HuggingFaceNPC":

        wrapped_llm = LangChainLLM(
            llm=llm,
            callback_manager=callback_manager
        )
        return cls(
            llm=wrapped_llm,
            **params
        )
    
    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))

        ai_message = self._llm.chat(chat_history).message
        chat_history.append(ai_message)

        return ai_message.content

    def _call_function(self, function_call: dict) -> ChatMessage:
        raise NotImplementedError("This method is not implemented yet.")

    def live_chat(self) -> None:
        raise NotImplementedError("This method is not implemented yet.")