# Reproduce experimental results

Before running the experiments, download FOSI's source code:
```bash
git clone https://github.com/hsivan/fosi
```
Let `fosi_root` be the root folder of the project on your local computer.
Make sure to add `fosi_root` to `PYTHONPATH` before running the experiments:
```bash
export PYTHONPATH=$PYTHONPATH:<fosi_root>
```

### Download external dataset

Download the external dataset used in the AC (MobileNetV1 on AudioSet data) experiment:
1. Change directory into the AudioSet dataset folder: `cd <fosi_root>/experiments/audioset_dataset`
2. Download the train_wav folder and train.csv file from https://www.kaggle.com/datasets/zfturbo/audioset
3. Download the valid_wav folder and valid.csv file from https://www.kaggle.com/datasets/zfturbo/audioset-valid
4. Run the script that converts the wav files to melspectogram images: `python convert_to_melspectogram.py`

The script runs for ~1 hour and creates two folders, `train_jpg` and `valid_jpg`, with melspectogram images.


### Installation

Running the experiments could be done using a Docker container or directly.
If using Docker container, ignore the following installation instructions and see [instructions for Docker](#run-as-a-docker-container).
Otherwise, CUDA toolkit and other requirements must be installed manually.

#### CUDA toolkit

To run FOSI with GPU, CUDA toolkit must be installed.
If using conda environment, the installation command is:
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda
```
Otherwise, a global installation is required:
```bash
sudo apt-get install cuda
```
After installing CUDA toolkit, follow [NVIDIA's environment setup instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup)
to set the environment variables PATH and LD_LIBRARY_PATH.
To find the lib/bin folders in case of conda environment use `find ~ -name 'libcusolver.so.11'` and in case of a
global installation with apt-get `find /usr/ -name 'libcusolver.so.11'` and use the containing folder.

Note: CUDA toolkit installation is not required when using the Docker container to run the experiments, or if running on the CPU.


#### Experiment requirements

To install the requirements for the experiments run:
```bash
cd <fosi_root>
pip install -r experiments/experiments_requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Quadratic functions
In these experiments we minimize quadratic functions.
Run these experiments from within the `experiments/quadratic` folder:
```bash
cd <fosi_root>/experiments/quadratic
```

Run the following three scripts:
```bash
python quadratic_jax_random_ortho_basis_gd.py
python quadratic_jax_random_ortho_basis.py
python quadratic_jax_kappa_zeta.py
```

After running all the experiments, run the following to generate the figures:
```bash
python plot_quadratic.py
```
The figures can be found under `<fosi_root>/experiments/quadratic/figures`.


## Deep neural networks
In these experiments we train DNNs with standard datasets.
Run these experiments from within the `experiments/dnn` folder:
```bash
cd <fosi_root>/experiments/dnn
```

Run the following three scripts (can run in parallel to save time):
```bash
python logistic_regression_mnist.py
python transfer_learning_cifar10.py
python autoencoder_cifar10.py
python rnn_shakespeare.py
python mobilenet_audioset.py
```

After running all the experiments, run the following to generate the figures:
```bash
python plot_dnn.py
```
The figures can be found under `<fosi_root>/experiments/dnn/figures`.

To generate loss and accuracy summary tables run:
```bash
python generate_result_summary.py
```
The script generates four `*_summary.csv` files with the relevant information.


## Run as a docker container

We provide Dockerfile to support building the project as a docker image.
To build the docker image you must first install docker engine and docker cli,
and set up the [Nvidia container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).
After installing these, run the command to build the docker image from within <fosi_root>:
```bash
cd <fosi_root>
sudo docker build -f experiments/experiments.Dockerfile -t fosi_experiment .
```
This docker image could be used to run the different experiments.

### Quadratic functions
The docker supports running the experiments `quadratic_jax_random_ortho_basis_gd`, `quadratic_jax_random_ortho_basis`, and `quadratic_jax_kappa_zeta`.
For example, to run the `quadratic_jax_random_ortho_basis_gd` execute the following commands
```bash
cd <fosi_root>
export experiment=quadratic_jax_random_ortho_basis_gd
export local_result_dir=$(pwd)"/experiments/quadratic/test_results"
export docker_result_dir="/app/experiments/test_results"
sudo docker run --gpus all -v ${local_result_dir}:${docker_result_dir} --rm fosi_experiment python3 /app/experiments/quadratic/${experiment}.py
```

The result folders and files can be found in the same location as running the experiments without Docker, under `<fosi_root>/experiments/quadratic`.

### Deep neural networks
The docker supports running the experiments `logistic_regression_mnist`, `transfer_learning_cifar10`, `autoencoder_cifar10`, `rnn_shakespeare`, and `mobilenet_audioset`.
For example, to run the `logistic_regression_mnist` execute the following commands
```bash
cd <fosi_root>
export experiment=logistic_regression_mnist
export local_result_dir=$(pwd)"/experiments/dnn/test_results_"${experiment}
export docker_result_dir="/app/experiments/test_results_"${experiment}
sudo docker run --gpus all -v ${local_result_dir}:${docker_result_dir} --rm fosi_experiment python3 /app/experiments/dnn/${experiment}.py
```

The result folders and files can be found in the same location as running the experiments without Docker, under `<fosi_root>/experiments/dnn`.
