import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment

# Step 0: Convert MP3 files to WAV format
def convert_mp3_to_wav(root_dir):
    for speaker in os.listdir(root_dir):
        speaker_dir = os.path.join(root_dir, speaker)
        for file_name in os.listdir(speaker_dir):
            if file_name.endswith(".mp3"):
                mp3_path = os.path.join(speaker_dir, file_name)
                wav_path = os.path.join(speaker_dir, f"{os.path.splitext(file_name)[0]}.wav")
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(wav_path, format="wav")
                print(f"Converted {mp3_path} to {wav_path}")
