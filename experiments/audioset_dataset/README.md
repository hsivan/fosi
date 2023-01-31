# Prepare AudioSet Dataset

Instructions for preparing the AudioSet dataset for train_mobilenet_on_audioset.py.

1. Download train_wav folder and train.csv file from https://www.kaggle.com/datasets/zfturbo/audioset
2. Download valid_wav folder and valid.csv file from https://www.kaggle.com/datasets/zfturbo/audioset-valid
3. Run the script convert_to_melspectogram.py

The scripted creates two folders, train_jpg and valid_jpg, with melspectogram images.