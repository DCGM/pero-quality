FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends git libsm6 libxext6 libxrender-dev
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir opencv-python numpy scipy shapely scikit-image lxml sklearn pyamg torch torchvision

# RUN git clone https://github.com/DCGM/pero-quality.git

