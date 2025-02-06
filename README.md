# Speaker Recognition System

A deep learning-based speaker recognition system built with PyTorch. This system can identify speakers from audio recordings using MFCC (Mel-frequency cepstral coefficients) features and a neural network classifier.

## Features

- Automatic conversion of MP3 files to WAV format
- MFCC feature extraction from audio files
- Neural network-based speaker classification
- Support for both training new models and using pre-trained models
- Real-time speaker prediction with confidence scores
- GPU acceleration support (when available)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/speaker-recognition.git
cd speaker-recognition
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg (required for audio processing):
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from ffmpeg website or use chocolatey: `choco install ffmpeg`

## Project Structure

```
speaker-recognition/
├── models/                 # Directory for saved models
├── data/                   # Directory for training data
│   ├── speaker1/          # Audio files for speaker 1
│   ├── speaker2/          # Audio files for speaker 2
│   └── ...
├── speaker_recognition/    # Main package
│   ├── __init__.py
│   ├── dataset.py         # Dataset handling
│   ├── model.py           # Neural network architecture
│   ├── trainer.py         # Training logic
│   └── predictor.py       # Prediction functionality
├── main.py                # Main script
├── utils.py               # Utility functions
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Usage

### Preparing the Data

1. Create a data directory structure where each speaker has their own subdirectory:
```
data/
├── speaker1/
│   ├── recording1.mp3
│   └── recording2.mp3
├── speaker2/
│   ├── recording1.mp3
│   └── recording2.mp3
└── ...
```

2. The system will automatically convert MP3 files to WAV format during training.

### Training a New Model

```python
from speaker_recognition import SpeakerDataset, SpeakerClassifier, ModelTrainer
from main import train_new_model

# Train a new model
model, label_encoder = train_new_model('data/', 'models/speaker_classifier.pth')
```

### Using a Pre-trained Model

```python
from main import load_model, SpeakerPredictor

# Load the model
model, label_encoder = load_model(
    'models/speaker_classifier.pth',
    'models/label_encoder.pth'
)

# Create predictor
predictor = SpeakerPredictor(model, label_encoder)

# Make prediction
result = predictor.predict('path/to/audio.wav')
print(f"Predicted Speaker: {result['speaker']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Running the Main Script

```bash
python main.py
```

## Model Architecture

The speaker recognition system uses a neural network with the following architecture:
- Input layer: Flattened MFCC features
- Hidden layers with ReLU activation and dropout
- Output layer: Number of speakers (classes)

## Requirements

- Python 3.8+
- PyTorch 2.1.0
- TorchAudio 2.1.0
- NumPy 1.24.3
- scikit-learn 1.3.1
- pydub 0.25.1
- FFmpeg (system requirement)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Built with PyTorch and TorchAudio
- MFCC feature extraction based on standard audio processing techniques
- Neural network architecture inspired by modern speaker recognition systems