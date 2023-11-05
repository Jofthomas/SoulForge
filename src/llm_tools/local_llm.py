import torch
from transformers import (
    BitsAndBytesConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    )
from langchain.llms import HuggingFacePipeline
from typing import Any, List, Mapping, Optional

QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

DEFAULT_BATCH_SIZE = 4

class QuantizedHFPipe(HuggingFacePipeline):
    """Langchain LLM wrapper for HF pipeline with quantization support.

    Example:

    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

    llm = CustomHFPipe.from_model_id_q(
        model_id=MODEL_NAME,
        task="text-generation",
        model_kwargs={"temperature": 0.01, "max_length": 296, 'do_sample': True},
    )

    TODO add passing optional quantization config
    TODO add streaming and async methods
    """
    # task: str
    device_map: Any
    quantization_config: Any
    
    @classmethod
    def from_model_id_q(
        cls,
        model_id: str,
        task: str,
        device: int = -1,
        model_kwargs: Optional[dict] = None,
        pipeline_kwargs: Optional[dict] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ) -> HuggingFacePipeline:
        
        _model_kwargs = model_kwargs or {}

        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
        model_quantized = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=QUANTIZATION_CONFIG,
            device_map="auto",
            **_model_kwargs
            )
        model_quantized.eval()
        
        from transformers import pipeline as hf_pipeline
        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }
        _pipeline_kwargs = pipeline_kwargs or {}
        
        if task == "text-generation":
            pipeline = hf_pipeline(
                task=task,
                model=model_quantized,
                tokenizer=tokenizer,
                # generation_config=gen_cfg,
                device_map="auto",
                # return_full_text=False,
                # return_full_text=True,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                batch_size=batch_size,
                model_kwargs=_model_kwargs,
                **_pipeline_kwargs,
            )
            pipeline.tokenizer.pad_token_id = model_quantized.config.eos_token_id
            pipeline.tokenizer.eos_token_id = model_quantized.config.eos_token_id
        else:
            raise ValueError(
                f"Got invalid task {task}"
            )
        return cls(
            pipeline=pipeline,
            model_id=model_id,
            model_kwargs=_model_kwargs,
            pipeline_kwargs=_pipeline_kwargs,
            batch_size=batch_size,
            **kwargs,
        )