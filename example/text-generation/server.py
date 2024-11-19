import asyncio
from fastapi import FastAPI, HTTPException, Request
from vllm import LLM, SamplingParams, CompletionOutput
import dataclasses
from typing import List

app = FastAPI()

# Load the model with tensor parallelism set to 4
import torch

model = LLM("Qwen/Qwen2.5-32B-Instruct", tensor_parallel_size=4, max_model_len=8192, max_seq_len_to_capture=8192, enable_prefix_caching=True, device="tpu", dtype=torch.bfloat16, max_num_seqs=256)

def req_out_to_dict(x):
    return {
        'outputs': list(map(dataclasses.asdict, x.outputs))
    }

@app.post("/batch")
async def generate(request: Request):
    data = await request.json()
    prompts = data.get("prompts")
    if not prompts:
        raise HTTPException(status_code=400, detail="Prompts are required")
    # Create a future for the request
    sampling_params = SamplingParams(temperature=data['samplings']['temperature'], max_tokens=data['samplings']['max_tokens'], min_p=data['samplings']['min_p'])
    results = model.generate(prompts, sampling_params)

    return {"result": list(map(req_out_to_dict, results))}

@app.get('/heartbeat')
async def heartbeat():
    return "Hi!"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, ssl_certfile="./ssh.crt.pem", ssl_keyfile="./ssh.key.pem", ssl_keyfile_password='Your password')

