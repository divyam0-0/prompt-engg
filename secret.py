# from langchain.prompts import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     AIMessagePromptTemplate
# )
# import langchain.chains
# import os 
# from langchain_core.prompts import ChatPromptTemplate
# from transformers import AutoTokenizer, AutoModelForCausalLM  # Changed import
# from langchain_community.llms import HuggingFacePipeline  # New import
# import torch

# # Initialize DeepSeek model instead of Groq
# model_name = "deepseek-ai/deepseek-llm-7b-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
# pipe = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     temperature=0,
#     max_new_tokens=512
# )
# llm = HuggingFacePipeline(pipeline=pipe)  # Changed LLM initialization

# # Everything below this line remains EXACTLY the same
# import mlflow # type: ignore
# mlflow.set_tracking_uri(uri="http://localhost:8505/")
# # ... rest of original code continues unchanged ...