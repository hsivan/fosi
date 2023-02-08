FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04

# Prevents debconf warning about delaying package configuration
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NOWARNINGS="yes"

# Install python3-pip
RUN apt-get update && apt-get install python3-pip -y

# Install cuda-nvcc that includes ptxas, which JAX requires and is not included in the installed CUDA toolkit.
# The version should match the CUDA toolkit used (11.4).
RUN apt-get install -y cuda-nvcc-11-4
# RUN echo $(find /usr/ -name 'ptxas')

# Install FOSI's requirements
COPY experiments/experiments_requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install fosi package
COPY . /app
RUN pip3 install /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

WORKDIR "/app/experiments"
CMD python3 dnn/logistic_regression_mnist.py