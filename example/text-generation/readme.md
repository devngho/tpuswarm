## text-generation

This example demonstrates how to use vLLM with tpuswarm to run a text generation task on a TPU. It uses miniconda, fastapi and vLLM to run a text generation task on a TPU.
 
You can use models up to ~=50B(bf16) on TPU v4-8.

### Usage

Create a (secret) gists containing the `server.py` script and your self-signed certificate.
Then modify the `setup.sh` script to download the files from the gists.
Finally, run the following command to start the TPU instances and run the sequence classification task.

```bash
python tpuswarm.py --region=us-central2-b --project=your-project --tpu-device=v4-8 --node-count=4 --batch=512 --command="screen -L -Logfile logfile.txt -d -m bash -c \"bash <(curl -sL [your setup.sh])\"" --port=5000 --host=0.0.0.0
```