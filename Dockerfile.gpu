FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

RUN apt-get update \
    && apt-get install -y python3-dev python3 python3-pip nano bash wget curl \
                         cuda-cublas-10.0 cuda-cufft-10.0 cuda-curand-10.0 cuda-cusolver-10.0 \
                         cuda-cusparse-10.0 cuda-cudart-10.0

WORKDIR /app
COPY . .

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt

# xRUN python3 test.py

# RUN python3 src/models.py
CMD ["/bin/bash"]
