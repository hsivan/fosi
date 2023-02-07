FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04

# Install python3-pip
RUN apt update && apt install python3-pip -y

COPY experiments/experiments_requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY . /app
RUN pip3 install /app

RUN pip3 install nvidia-pyindex
RUN pip3 install nvidia-cuda-nvcc-cu114
# RUN echo $(find /usr/ -name 'ptxas')

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/lib/python3.8/dist-packages/nvidia/cuda_nvcc/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV PYTHONPATH "${PYTHONPATH}:/app"

WORKDIR "/app/experiments"
CMD python3 dnn/logistic_regression_mnist.py