FROM docker.io/tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends git libsm6 libxext6 libxrender-dev

RUN git clone https://github.com/DCGM/pero-quality.git

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r pero-quality/requirements.txt

WORKDIR pero-quality

