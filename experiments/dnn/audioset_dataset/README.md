# External dataset

Instructions for preparing the dataset for the AC (MobileNetV1 on AudioSet data) experiment (`train_mobilenet_on_audioset.py`).

1. Cd into the `audioset_dataset` folder: `cd <fosi_root>/experiments/audioset_dataset`
2. Download the train_wav folder, class_labels_indices.csv file, and train.csv file from https://www.kaggle.com/datasets/zfturbo/audioset
3. Download the valid_wav folder and valid.csv file from https://www.kaggle.com/datasets/zfturbo/audioset-valid
4. Run the script `convert_to_melspectogram.py`: `python convert_to_melspectogram.py`

The script runs for ~1 hour and creates two folders, `train_jpg` and `valid_jpg`, with melspectogram images.