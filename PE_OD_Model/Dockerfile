FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04
WORKDIR /worker-model
RUN mkdir /miniconda3
COPY ./requirements.txt /
COPY ./worker-model/* /worker-model

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    g++
RUN \
    mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p /miniconda3 && \ 
    rm ~/miniconda3/miniconda.sh

ENV PATH="/miniconda3/bin:$PATH"
RUN conda install python=3.8.20 -y
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN pip install --no-cache-dir -r /requirements.txt
RUN \ 
    pip install --no-cache-dir -r /requirements.txt && rm /requirements.txt && \
    mim install mmengine==0.10.3 && \
    mim install mmcv==2.0.1 && mim install mmdet==3.1.0 && mim install mmpose==1.3.1

# CMD ["conda", "run", "--no-capture-output", "-n", "base", "python", "/worker-model/detection.py"]
