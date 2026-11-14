"""Optional no-key, no-quota text reasoning on Kaggle GPU0.

Qwen3-1.7B Apache 2.0; output streams token fragments with non-thinking mode.
This is slower/smaller than hosted DeepSeek and competes with ASR on GPU0.
"""
import asyncio
from functools import lru_cache
import os
import threading

@lru_cache(maxsize=1)
def _models():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name=os.getenv("LOCAL_LLM_MODEL","Qwen/Qwen3-1.7B")
    gpu=os.getenv("LOCAL_LLM_DEVICE","cuda:0")
    tokenizer=AutoTokenizer.from_pretrained(name)
    model=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float16).to(gpu)
    model.eval()
    return model,tokenizer

async def stream_local(messages):
    import torch
    from transformers import TextIteratorStreamer,StoppingCriteria,StoppingCriteriaList
    model,tokenizer=await asyncio.to_thread(_models)
    text=tokenizer.apply_chat_template(messages,tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    inputs=tokenizer([text],return_tensors="pt").to(model.device)
    stop=threading.Event()
    streamer=TextIteratorStreamer(tokenizer,skip_prompt=True,
        skip_special_tokens=True,timeout=60)
    errors=[]
    class StopIfCanceled(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            return stop.is_set()
    def run():
        try:
            with torch.inference_mode():
                model.generate(**inputs,streamer=streamer,
                    max_new_tokens=int(os.getenv("LOCAL_LLM_MAX_TOKENS","180")),
                    do_sample=True,temperature=0.7,top_p=0.8,
                    stopping_criteria=StoppingCriteriaList([StopIfCanceled()]),
                    pad_token_id=tokenizer.eos_token_id)
        except Exception as exc:
            errors.append(exc)
            streamer.end()
    worker=threading.Thread(target=run,daemon=True)
    worker.start()
    itr=iter(streamer)
    def next_chunk():
        try:return next(itr)
        except StopIteration:return None
    try:
        while True:
            chunk=await asyncio.to_thread(next_chunk)
            if chunk is None:break
            if chunk:yield chunk
        if errors:raise RuntimeError("Local model: "+str(errors[0])[:250])
    finally:
        stop.set()
