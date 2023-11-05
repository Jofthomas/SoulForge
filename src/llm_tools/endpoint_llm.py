import requests
from typing import (
    Any, 
    Dict, 
    List, 
    Mapping, 
    Optional,
    Iterator,
)
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.output import GenerationChunk
from langchain.llms.base import LLM
from langchain.schema import (
    Generation,
    LLMResult,
)

def _stream_response_to_generation_chunk(
    stream_response: str,
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    return GenerationChunk(
        text=stream_response,
        generation_info={}
    )

class EndpointLLM(LLM):
    """Langchain wrapper for endpoint with LLM inference server.

    Supports streaming or sending full response directly
    TODO add async methods
    """
    api_endpoint: str
    endpoint_path: str = "/v1/models/model:predict"
    streaming: bool = False
    max_new_tokens: Optional[int] = 296
    temperature: Optional[float] = 0.5

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = [],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        data = {"prompt": prompt,
                "temperature": kwargs.get('temperature', self.temperature),
                "max_new_tokens": kwargs.get('max_new_tokens', self.max_new_tokens),
                "stop": stop or [],
                }
        data.update({param: value for param, value in kwargs.items() if param not in data})
        try:
            response = requests.post(
                self.api_endpoint + self.endpoint_path,
                json=data,
            )
            if response.status_code == 200:
                text = dict(response.json())['data']['generated_text']
            else:
                raise ValueError(f'The response status code was: {response.status_code}, '
                                 'expected: 200')
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)
        
        return text
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        data = {
            "prompt": prompt,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_new_tokens": kwargs.get('max_new_tokens', self.max_new_tokens),
            "stop": stop or [],
            "stream": True,
            }
        data.update({param: value for param, value in kwargs.items() if param not in data})
        try:
            response = requests.post(
                self.api_endpoint + self.endpoint_path,
                json=data,
                stream=True,
            )
            if response.status_code == 200:
                for stream_resp in response.iter_content(chunk_size=1024):
                    if stream_resp:
                        chunk = _stream_response_to_generation_chunk(stream_resp)
                        yield chunk
                        if run_manager:
                            run_manager.on_llm_new_token(
                                chunk.text,
                                chunk=chunk,
                                verbose=self.verbose,
                                logprobs=chunk.generation_info["logprobs"]
                                if chunk.generation_info
                                else None,
                            )
            else:
                raise ValueError(f'The response status code was: {response.status_code}, '
                                'expected: 200')
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

    
    def _generate(self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        if len(prompts) > 1:
            raise NotImplementedError("Multiple prompts are not supported for now.")
        if self.streaming:
            generation: Optional[GenerationChunk] = None
            for chunk in self._stream(prompts[0], stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return LLMResult(generations=[[generation]])

        full_response = self._call(
            prompt=prompts[0], 
            run_manager=run_manager, 
            **kwargs
        )

        return LLMResult(
            generations=[
                [Generation(text=full_response)]
            ]
        )
    
    def create_llm_result(
        self, choices: Any, prompts: List[str], token_usage: Dict[str, int]
    ) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        generations = []
        for i, _ in enumerate(prompts):
            generations.append(
            
                Generation(
                    text=choices[i],
                    generation_info={}
                )
                    
            )
        llm_output = {"token_usage": token_usage, "model_name": "custom"}
        return LLMResult(generations=generations, llm_output=llm_output)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"endpoint": self.api_endpoint}