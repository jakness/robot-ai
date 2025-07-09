# robot-ai

## Setup the environment

Create a conda virtual environment and activate it, e.g. with [`miniconda`](https://www.anaconda.com/docs/getting-started/miniconda/main):
```bash
conda create -y -n robot-ai python=3.10
conda activate robot-ai
```

Install `ffmpeg`:
```bash
  conda install ffmpeg -c conda-forge
```

Install `PyTorch`:
```bash
  conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

Install `robot-ai`:
```bash
  pip install -e ".[dev]"
```

Install the pre-commit hooks
```bash
   pre-commit install
```
