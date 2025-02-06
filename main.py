import torch
from pathlib import Path
from speaker_recognition import SpeakerDataset, SpeakerClassifier, ModelTrainer, SpeakerPredictor
from utils import convert_mp3_to_wav

def train_new_model(data_dir, model_save_path='models/speaker_classifier.pth'):
    # Convert MP3 files to WAV format
    convert_mp3_to_wav(data_dir)

    # Create dataset
    dataset = SpeakerDataset(data_dir, n_mfcc=13, n_mels=40)
    
    # Initialize model
    sample_mfcc, _ = dataset[0]
    input_size = sample_mfcc.numel()
    num_classes = len(dataset.speakers)
    model = SpeakerClassifier(input_size, num_classes)
    
    # Train model
    trainer = ModelTrainer(model)
    trainer.train(dataset, batch_size=32, num_epochs=20)
    
    # Save model and label encoder
    Path('models').mkdir(exist_ok=True)
    trainer.save_model(model_save_path)
    torch.save(dataset.label_encoder, 'models/label_encoder.pth')
    
    return model, dataset.label_encoder

def load_model(model_path, label_encoder_path):
    label_encoder = torch.load(label_encoder_path)
    
    # Initialize model with correct dimensions
    num_classes = len(label_encoder.classes_)
    model = SpeakerClassifier(input_size=13*40, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    
    return model, label_encoder

def main():
    DATA_DIR = 'data/'
    MODEL_PATH = 'models/speaker_classifier.pth'
    LABEL_ENCODER_PATH = 'models/label_encoder.pth'
    
    if not Path(MODEL_PATH).exists():
        model, label_encoder = train_new_model(DATA_DIR)
    else:
        model, label_encoder = load_model(MODEL_PATH, LABEL_ENCODER_PATH)
    
    # Create predictor
    predictor = SpeakerPredictor(model, label_encoder)
    
    # Example prediction
    test_file = "test_data/test_1.wav"
    result = predictor.predict(test_file)
    print(f"Predicted Speaker: {result['speaker']}")
    print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    main()
