import json
from llama_index.llms import OpenAI, ChatMessage
from llama_index.tools import BaseTool
from typing import (
    Sequence,
    List,
    Optional,
)

from config import *
from .base import BaseNPC
from ..personality import Personality

import nest_asyncio

nest_asyncio.apply()


class OpenAINPC(BaseNPC):

    def __init__(
            self, 
            llm: OpenAI = OpenAI(temperature=0, model="gpt-3.5-turbo-0613", max_tokens=100),
            actions: Sequence[BaseTool] = [],
            chat_history: List[ChatMessage] = [],
            personality: Optional[str | Personality] = None,
        ) -> None:
        
        self._llm = llm
        self._actions = {tool.metadata.name: tool for tool in actions}
        
        # Add the personality message to chat history if personality is provided
        if personality is not None:
            chat_history.insert(0, ChatMessage(content=personality.get_description(), role="system"))
        self._chat_history = chat_history
        self._personality = personality

    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))
        functions = [
            tool.metadata.to_openai_function() for _, tool in self._actions.items()
        ]
        ### retrive infos from memory
        
        
        ### send infos to LLM
        
        if len(functions) == 0:
            ai_message = self._llm.chat(chat_history).message
        else:
            ai_message = self._llm.chat(chat_history, functions=functions).message
        chat_history.append(ai_message)

        function_call = ai_message.additional_kwargs.get("function_call", None)
        if function_call is not None:
            function_message = self._call_function(function_call)
            chat_history.append(function_message)
            ai_message = self._llm.chat(chat_history).message
            chat_history.append(ai_message)

        return ai_message.content

    def _call_function(self, function_call: dict) -> ChatMessage:
        tool = self._actions[function_call["name"]]
        output = tool(**json.loads(function_call["arguments"]))
        return ChatMessage( # <- Note we're returning the entire ChatMessage object
            name=function_call["name"],
            content=str(output),
            role="function",
            additional_kwargs={"name": function_call["name"]},
        )

    def live_chat(self) -> None:
        print("You are now chatting with the NPC. Type 'exit()' to end the conversation.")
        while True:
            # Get player input
            player_input = input("You: ")
            if player_input.lower() == 'exit()':
                print("You ended the conversation.")
                break
            chat_history = self._chat_history
            chat_history.append(ChatMessage(role="user", content=player_input))
            functions = [
                tool.metadata.to_openai_function() for _, tool in self._actions.items()
            ]
            ### retrive infos from memory
        
        
            ### send infos to LLM
            ai_message = self._llm.chat(chat_history, functions=functions).message
            chat_history.append(ai_message)
            
            function_call = ai_message.additional_kwargs.get("function_call", None)
        
            if function_call is not None:
                function_message = self._call_function(function_call)
                
                chat_history.append(function_message)
                if function_message.content == 'exit()':
                    print(f"{self._personality.name} ended the conversation.")
                    break

                ai_message = self._llm.chat(chat_history).message
                chat_history.append(ai_message)
            print(ai_message.content)
