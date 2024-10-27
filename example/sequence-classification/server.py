import asyncio
from fastapi import FastAPI, HTTPException, Request

import dataclasses
from typing import List
from transformers import FlaxAutoModelForSequenceClassification, AutoTokenizer
import jax
import jax.numpy as jnp
import numpy as np

app = FastAPI()

model_id = 'devngho/code_edu_classifier_v2_microsoft_codebert-base'
model = FlaxAutoModelForSequenceClassification.from_pretrained(model_id, dtype=jnp.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model_func = jax.pmap(lambda x, mask: model(x, mask).logits)

@app.post("/batch")
async def generate(request: Request):
    data = await request.json()
    prompts = data.get("prompts")
    if not prompts:
        raise HTTPException(status_code=400, detail="Prompts are required")

    tok = tokenizer(prompts, padding='max_length', truncation=True, max_length=512, return_tensors='jax')

    input_ids = jnp.array(jnp.array_split(tok['input_ids'], jax.device_count()))
    attention_mask = jnp.array(jnp.array_split(tok['attention_mask'], jax.device_count()))

    logits = list(map(lambda x: x[0], np.concatenate(np.array(model_func(input_ids, attention_mask)), axis=0).tolist()))

    return {"result": logits}

@app.get('/heartbeat')
async def heartbeat():
    return "Hi!"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, ssl_certfile="./ssl.crt.pem", ssl_keyfile="./ssl.key.pem", ssl_keyfile_password='Your password')

