curl --output ssl.crt.pem "Your gist url"
curl --output ssl.key.pem "Your gist url"
curl --output embeddings.py "Your gist url"

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

. ~/miniconda3/etc/profile.d/conda.sh

conda create -n myenv python=3.10 -y
conda activate myenv

sudo DEBIAN_FRONTEND=noninteractive apt update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install libopenblas-base libopenmpi-dev libomp-dev -y

# Install JAX and Pallas.
pip install 'transformers[flax]'
pip install jax[tpu]==0.4.35 flax~=0.10.0 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install tokenizers fastapi
pip install 'uvicorn[standard]'
pip install git+https://github.com/devngho/transformers@3815933c98a48a25c81fd48836502dbef1573617

export HF_TOKEN="Your HF token"

python embeddings.py

