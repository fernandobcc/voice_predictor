import torch
import torchaudio

class SpeakerPredictor:
    def __init__(self, model, label_encoder, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.label_encoder = label_encoder
        self.device = device
        self.model.to(device)
        self.model.eval()

    def extract_mfcc(self, file_path, n_mfcc=13, n_mels=40):
        waveform, sample_rate = torchaudio.load(file_path)
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={'n_mels': n_mels}
        )
        mfcc = mfcc_transform(waveform)
        return mfcc.mean(dim=2).squeeze()

    def predict(self, file_path):
        mfcc = self.extract_mfcc(file_path).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(mfcc)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            speaker_name = self.label_encoder.inverse_transform([predicted.item()])[0]
            
            return {
                'speaker': speaker_name,
                'confidence': confidence.item()
            }