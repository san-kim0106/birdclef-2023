from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC, Wav2Vec2Model
# from datasets import laod_dataset
import datasets
import torch
import torch.nn as nn

class Wav2VecAudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        self.pooling = nn.AvgPool1d(768)

        self.dropout = nn.Dropout(0.5)

        self.projection = nn.Linear('''Number of encoding tokens''', '''Number of classes''')

        self.relu = nn.ReLU()
    
    def forward(self, audio):
        context_representation = self.wav2vec(audio)[0]
        pooling_state = self.pooling(context_representation)
        logits = self.relu(self.projection(pooling_state))

        return logits

