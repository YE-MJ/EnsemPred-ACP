import torch
import numpy as np
import joblib
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
import seaborn as sns
import torch.nn.functional as F
import torch.nn as nn


class SequenceModel_deep(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(SequenceModel_deep, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim*2)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.bn1(lstm_out)
        x = F.relu(self.fc1(lstm_out))
        x = self.bn2(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output

def calculate_binary_profile_features_with_fixed_length(sequence, fixed_length=1000):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_binary = {aa: np.array([1 if amino_acids[i] == aa else 0 for i in range(len(amino_acids))]) for aa in amino_acids}

    if len(sequence) > fixed_length // 20:
        sequence = sequence[:fixed_length // 20]
    
    binary_profile = [aa_to_binary.get(aa, np.zeros(20)) for aa in sequence]

    if len(binary_profile) < fixed_length // 20:
        binary_profile.extend([np.zeros(20)] * (fixed_length // 20 - len(binary_profile)))
    return np.array(binary_profile).reshape(-1)


def ensemble_prott5_LSTM(df, seq, fixed_length=1000):
    X_prott5 = df.drop(['SampleName'], axis=1)
    print(X_prott5.shape)
    sequences = seq

    bpf_features_new = [calculate_binary_profile_features_with_fixed_length(s, fixed_length) for s in sequences]
    bpf_features_padded_new = np.array(bpf_features_new)
    
    X_combined_new = np.hstack((X_prott5, bpf_features_padded_new))

    scaler = joblib.load('./Prott5/LSTM(deep)_Prott5+BPF_weights/scaler_LSTM(deep)_Prott5+BPF.pkl')
    X_combined_scaled_new = scaler.fit_transform(X_combined_new)

    X_tensor_new = torch.tensor(X_combined_scaled_new, dtype=torch.float)

    # Test 
    best_model_paths_acc = [
        './Prott5/LSTM(deep)_Prott5+BPF_weights/best_model_fold_1_acc.pth',
        './Prott5/LSTM(deep)_Prott5+BPF_weights/best_model_fold_2_acc.pth',
        './Prott5/LSTM(deep)_Prott5+BPF_weights/best_model_fold_3_acc.pth',
        './Prott5/LSTM(deep)_Prott5+BPF_weights/best_model_fold_4_acc.pth',
        './Prott5/LSTM(deep)_Prott5+BPF_weights/best_model_fold_5_acc.pth'
    ]

    input_dim = X_combined_scaled_new.shape[1]
    hidden_dim = 128
    output_dim = 2

    all_predictions = []

    # 각 fold별 모델 로드 및 예측 (best - acc)
    for model_path in best_model_paths_acc:
        model = SequenceModel_deep(input_dim, hidden_dim, output_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_dataset = TensorDataset(X_tensor_new)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        y_preds = []

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0]
                if isinstance(inputs, list):
                    inputs = torch.tensor(inputs, dtype=torch.float) 
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                positive_probabilities = probabilities[:, 1].numpy()
                y_preds.extend(positive_probabilities)

        all_predictions.append(y_preds)

    # Ensemble the predictions by averaging
    ensemble_predictions = np.mean(all_predictions, axis=0)
    return ensemble_predictions
