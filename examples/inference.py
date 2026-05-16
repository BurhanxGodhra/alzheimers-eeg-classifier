"""
Inference example: Load a trained EEGNet model and predict on a new EEG file.
"""

import torch
import mne
import numpy as np
import sys
sys.path.append('..')
from src.models import EEGNet

def load_and_preprocess_eeg(file_path, sfreq_target=250, tmin=0, tmax=2):
    """Load an EEG file and return a preprocessed epoch (ready for model)."""
    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
    raw.apply_function(lambda x: (x - np.mean(x)) / np.std(x), channel_wise=True)
    raw.filter(0.5, 45.0, fir_design='firwin', verbose=False)
    raw.notch_filter(50.0, fir_design='firwin', verbose=False)
    raw.resample(sfreq_target, verbose=False)
    
    start = raw.times[0]
    end = start + 2.0
    raw_crop = raw.copy().crop(start, end)
    data = raw_crop.get_data()[:, :500]  
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)  

def predict(model, eeg_tensor, device='cpu'):
    """Run inference and return predicted class and probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(eeg_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs.cpu().numpy()

if __name__ == '__main__':
    eeg_file = 'path/to/your/subject_eeg.set'
    model_path = '../experiments/results/eegnet/best_model.pth'
    model = EEGNet(in_channels=19, n_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    eeg_tensor = load_and_preprocess_eeg(eeg_file)
    
    
    pred_class, probs = predict(model, eeg_tensor)
    class_names = ['Normal', 'Impaired']
    print(f"Predicted: {class_names[pred_class]} (probabilities: Normal={probs[0][0]:.3f}, Impaired={probs[0][1]:.3f})")
