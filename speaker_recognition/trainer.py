import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

class ModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train(self, dataset, batch_size=32, num_epochs=20, learning_rate=0.001):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            self.model.train()
            for mfcc, labels in dataloader:
                mfcc, labels = mfcc.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(mfcc)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(dataloader)
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Loss: {epoch_loss:.4f}, "
                  f"Accuracy: {accuracy:.2f}%")
    
    def save_model(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)