
from scipy.io import loadmat
import torch.nn as nn
import os
import torch
from torch.utils.data import Dataset

class Dataset_PPMI(Dataset):
    def __init__(self, root_dir, win=1, spatial=False, graph=False, mgnn=False, file_paths=None):
        super(Dataset_PPMI, self).__init__()
        self.graph = graph
        self.spatial = spatial
        self.mgnn = mgnn
        self.win = win
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.window_indices = []
        self.data_path = []
        self.load_data(file_paths)
        self.softmax = nn.Softmax(dim=-1)

    def load_data(self, file_paths=None):
        sentence_sizes = []
        if file_paths is None:
            for subdir, _, files in os.walk(self.root_dir):
                for file in files:
                    if 'AAL116_features_timeseries' in file:
                        file_path = os.path.join(subdir, file)
                        self.data_path.append(file_path)
                        data = loadmat(file_path)
                        features = data['data']
                        sentence_sizes.append(features.shape[0])
                        label = self.get_label(subdir)
                        self.data.append(features)
                        self.labels.append(label)
                        num_windows = self.win
                        self.window_indices.extend([(len(self.data) - 1, w_idx) for w_idx in range(num_windows)])
        else:
            for file_path in file_paths:
                data = loadmat(file_path)
                features = data['data']
                sentence_sizes.append(features.shape[0])
                label = self.get_label(os.path.dirname(file_path))
                self.data.append(features)
                self.labels.append(label)
                num_windows = self.win
                self.window_indices.extend([(len(self.data) - 1, w_idx) for w_idx in range(num_windows)])
        if self.spatial == False:
            self.max_sentence_size = max(sentence_sizes)

    def get_label(self, subdir):
        if 'control' in subdir:
            return 0
        elif 'patient' in subdir:
            return 1
        elif 'prodromal' in subdir:
            return 2
        elif 'swedd' in subdir:
            return 3
        else:
            print("Label error")
            return -1

    def pad_sentences(self):
        self.data = [torch.cat(
            (torch.tensor(sentence), torch.zeros(self.max_sentence_size - sentence.shape[0], sentence.shape[1])), dim=0)
                     for sentence in self.data]

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):

        data_idx, window_idx = self.window_indices[idx]
        eeg_data = self.data[data_idx]
        eeg_data = torch.nan_to_num(torch.tensor(eeg_data, dtype=torch.float32))
        label = self.labels[data_idx]

        labels = torch.tensor(label, dtype=torch.int64)
        data = eeg_data.squeeze().T
        sfc = torch.nan_to_num(torch.corrcoef(data), 1e-6)

        return sfc, labels

