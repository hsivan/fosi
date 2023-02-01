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

### External dataset

Download the external dataset used in the AC (MobileNetV1 on AudioSet data) experiment:
1. Cd into the `audioset_dataset` folder: `cd <fosi_root>/experiments/audioset_dataset`
2. Download the train_wav folder and train.csv file from https://www.kaggle.com/datasets/zfturbo/audioset
3. Download the valid_wav folder and valid.csv file from https://www.kaggle.com/datasets/zfturbo/audioset-valid
4. Run the script `convert_to_melspectogram.py`: `python convert_to_melspectogram.py`

The script runs for ~1 hour and creates two folders, `train_jpg` and `valid_jpg`, with melspectogram images.




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
