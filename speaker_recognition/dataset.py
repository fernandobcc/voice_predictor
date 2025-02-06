import os
import torch
import torchaudio
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class SpeakerDataset(Dataset):
    def __init__(self, root_dir, n_mfcc=13, n_mels=80):
        self.root_dir = root_dir
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        
        self.file_paths, self.labels, self.speakers = self._load_dataset()
        self.label_encoder = self._create_label_encoder()
        
    def _load_dataset(self):
        file_paths = []
        labels = []
        speakers = os.listdir(self.root_dir)
        
        for speaker in speakers:
            speaker_dir = os.path.join(self.root_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
                
            for file_name in os.listdir(speaker_dir):
                if file_name.endswith(".wav"):
                    file_paths.append(os.path.join(speaker_dir, file_name))
                    labels.append(speaker)
                    
        return file_paths, labels, speakers
    
    def _create_label_encoder(self):
        label_encoder = LabelEncoder()
        label_encoder.fit(self.speakers)
        return label_encoder

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        waveform, sample_rate = torchaudio.load(file_path)
        mfcc = self._extract_mfcc(waveform, sample_rate)
        label_encoded = self.label_encoder.transform([label])[0]
        
        return mfcc, label_encoded
    
    def _extract_mfcc(self, waveform, sample_rate):
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={'n_mels': self.n_mels}
        )
        mfcc = mfcc_transform(waveform)
        return mfcc.mean(dim=2).squeeze()
