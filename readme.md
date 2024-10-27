## tpuswarm

Create spot TPU instances, then run a batched job on them.

This project was supported with Cloud TPUs from Google's TPU Research Cloud [(TRC)](https://sites.research.google/trc/about/).⚡

### Usage

Examples are in the `example` directory.

```bash
. login_gcp # for ssh-agent. You should run it first!
```

```bash
python tpuswarm.py --region=us-central2-b --project=your-project --tpu-device=v4-8 --node-count=4 --batch=512 --command="echo \"Hello, TPUs\!\" > /tmp/hello.txt" --port=5000 --host=0.0.0.0
```

```bash
python tpuswarm_clean.py --region=us-central2-b --project=your-project
```

### Batch shape

Your program should host a HTTP API at 8080, and accept POST requests with a JSON body of the following shape:

```json
POST /batch
{
  "prompts": [ // the list of prompts to process
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog."
  ],
  "samplings": { // the sampling parameters or any configuration you need
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.0,
    "length": 128
  }
}

GET /heartbeat
200 OK
```

And return a JSON response of the following shape:

```json
{
  "result": [
    // any shape you want
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog."
  ]
}
```

You can send a same shape of request to the `/batch` endpoint at the tpuswarm endpoint, and it will distribute the requests to the TPUs and return the results.
tpuswarm will split the requests into `batch` size, and send them to the TPUs in parallel.