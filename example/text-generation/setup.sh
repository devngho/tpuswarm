curl --output ssh.crt.pem "Your gist url"
curl --output ssh.key.pem "Your gist url"
curl --output tpu_vllm_gen.py "Your gist url"

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

. ~/miniconda3/etc/profile.d/conda.sh

conda create -n myenv python=3.10 -y
conda activate myenv

sudo DEBIAN_FRONTEND=noninteractive apt update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install libopenblas-base libopenmpi-dev libomp-dev -y

mkdir ~/vllm_setup
cd ~/vllm_setup

git clone https://github.com/vllm-project/vllm.git
cd ./vllm

# Clean up the existing torch and torch-xla packages.
pip uninstall torch torch-xla -y

# Install PyTorch and PyTorch XLA.
export DATE="20240828"
export TORCH_VERSION="2.5.0"
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-${TORCH_VERSION}.dev${DATE}-cp310-cp310-linux_x86_64.whl
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-${TORCH_VERSION}.dev${DATE}-cp310-cp310-linux_x86_64.whl

# Install JAX and Pallas.
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

pip install -r requirements-tpu.txt
pip install setuptools_scm

VLLM_TARGET_DEVICE="tpu" python setup.py develop

cd ~/

export HF_TOKEN="Your HF token"

python tpu_vllm_gen.py

