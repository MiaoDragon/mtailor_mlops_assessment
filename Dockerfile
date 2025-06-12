FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update
# Install Python 3.10 and system dependencies
RUN apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils \
    wget curl git build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Set python3.10 as default python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/local/bin/pip /usr/bin/pip

# Upgrade pip
RUN python -m pip install --upgrade pip

FROM base AS torchstage

# Install PyTorch 2.7.1 with CUDA 11.8
RUN pip install torch==2.7.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

FROM torchstage AS final

WORKDIR /workspace

COPY . .

# Dependencies
RUN pip install -r requirements.txt

# install model
RUN wget https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth

EXPOSE 8192

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8192"]