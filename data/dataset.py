import torch
from PIL import Image
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from librosa.feature import melspectrogram
import sklearn
import scipy
import os
from tqdm import tqdm

class DataSet(torch.utils.data.Dataset):
    def __init__(self, paths=None, labels=None, one_hot_encodings=None):
        self.paths = paths
        self.labels = labels
        self.one_hot_encodings = one_hot_encodings

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        wave_data, sample_rate = sf.read(path)
        y = self.one_hot(self.labels[index])

        return wave_data, y
    
    def one_hot(self, label):
        v = np.zeros(len(self.one_hot_encodings))
        v[self.one_hot_encodings[label]] = 1
        return v
    
    def num_classes(self):
        return len(self.labels)
    
    def load_dataset(self, path):
        '''
        data
        |_ birdclef-2023
           |_ train_audio (pwd)
              |_ abethr1
              |_ abhorti1
              ...
        |
        |_ melspectrogram_dataset
        '''

        if (os.path.isdir(path)):
            audio_paths = pd.DataFrame(columns=["path", "label"])

            for label in tqdm(os.listdir(path)):
                if (not os.path.isdir(f"{path}/{label}")): continue

                for audio_file in os.listdir(f"{path}/{label}"):
                    audio_paths.loc[audio_paths.shape[0]] = {"path": f"{path}/{label}/{audio_file}", "label": label}
            
            audio_paths.to_csv(f"{path}/audio_paths.csv", index=False)

        return audio_paths

if __name__ == "__main__":
    dataset = DataSet()
    df = dataset.load_dataset("/Users/kimsan/Desktop/Lecture-Materials/3-1/AIGS538-Deep-Learning/final-project/data/birdclef-2023/train_audio")
    print(df.shape)

