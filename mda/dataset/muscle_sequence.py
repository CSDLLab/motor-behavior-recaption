import glob
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class MuscleSequence(Dataset):
    def __init__(self, data_root, time_steps=16, dilation=1, reverse=False, normalized=True) -> None:
        super(MuscleSequence, self).__init__()
        self.data_root = data_root
        self.time_steps = time_steps
        self.dilation = dilation
        self.reverse = reverse
        self.data_list_x = []
        self.data_list_u = []
        self.seq_map = {}
        self.seq_begin = {}
        self.num_sequences = 0
        self.num_nodes = 0
        self.load_data(reverse)
        self.ventral_edges = [(14, 0), (0, 15), (15, 1), (1, 16), (16, 2), (2, 17), (17, 3), (3, 18), (18, 4), (4, 19), (19, 5), (5, 20), (20, 6), (6, 21), (22, 7), (7, 23), (23, 8), (8, 24), (24, 9), (9, 25), (25, 10), (10, 26), (26, 11), (11, 27), (27, 12), (12, 28), (28, 13), (13, 29), (14, 22), (15, 23), (16, 24), (17, 25), (18, 26), (19, 27), (20, 28), (21, 29), (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13), (22, 33), (33, 32), (14, 31), (31, 30), (32, 30), (33, 31)]
        self.dorsal_edges = []
        if normalized and self.num_sequences:
            self.normalize_data()
    
    def load_data(self, reverse=False):
        data_list_x = []
        data_list_u = []
        seq_map = {}
        seq_begin = {}
        num_sequences = 0
        with open(self.data_root, 'rb') as fp:
            dataset = pickle.load(fp)
        
        for i, data in enumerate(dataset):
            df = data['data']
            x_cols = [c for c in df.columns if c[2] != 'calcium']
            u_cols = [c for c in df.columns if c[2] == 'calcium']
            angle_cols = [c for c in df.columns if c[2] == 'angle']
            for c in angle_cols:
                df[c] = df[c].apply(lambda x: x + 180.0 if x < 0.0 else x)
            x = df[x_cols].values.reshape(len(df), -1, 5).transpose((2, 0, 1))
            u = df[u_cols].values[np.newaxis, :, :]
            self.num_nodes = x.shape[2]
            
            num_seq = x.shape[1] // self.dilation - self.time_steps
            # filter the data less than self.time_steps
            if num_seq > 0:
                seq_map.update({k: len(data_list_x) for k in range(num_sequences, num_sequences + num_seq)})
                seq_begin.update({k: num_sequences for k in range(num_sequences, num_sequences + num_seq)})
                num_sequences += num_seq
                if reverse:
                    data_list_u.append(x)
                    data_list_x.append(u)
                else:
                    data_list_u.append(u)
                    data_list_x.append(x)
        
        self.num_sequences = num_sequences
        self.seq_begin = seq_begin
        self.seq_map = seq_map
        self.data_list_x = data_list_x
        self.data_list_u = data_list_u

        if num_sequences > 0:
            self.muscle_mean = np.concatenate(data_list_x, axis=1).mean(axis=(1, 2))
            self.muscle_std = np.concatenate(data_list_x, axis=1).std(axis=(1, 2))
            self.intensities_mean = np.concatenate(data_list_u, axis=1).mean(axis=(1, 2))
            self.intensities_std = np.concatenate(data_list_u, axis=1).std(axis=(1, 2))
        else:
            raise ValueError("time_steps or dilation too large!")

    def generate_adjacent(self, num_hop=3):
        adjacency = np.zeros((num_hop, self.num_nodes, self.num_nodes))
        if 'ventral.pkl' in self.data_root.split('/')[-1]:
            for e in self.ventral_edges:
                adjacency[0, e[0], e[1]] += 1
        elif 'dorsal.pkl' in self.data_root.split('/')[-1]:
            for e in self.dorsal_edges:
                adjacency[0, e[0], e[1]] += 1
        adjacency[0] = adjacency[0] + np.eye(self.num_nodes)
        for i in range(num_hop - 1):
            adjacency[i+1] = self.normalize_digraph(adjacency[i] @ adjacency[0])
        return adjacency

    def normalize_digraph(self, A):
        D1 = np.sum(A, 0)
        num_nodes = A.shape[0]
        Dn = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            if D1[i] > 0:
                Dn[i, i] = D1[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD

    def normalize_data(self):
        for i in range(len(self.data_list_x)):
            self.data_list_x[i] = self.data_list_x[i] - self.muscle_mean[:, np.newaxis, np.newaxis]
            self.data_list_x[i] = self.data_list_x[i] / self.muscle_std[:, np.newaxis, np.newaxis]
            self.data_list_u[i] = self.data_list_u[i] - self.intensities_mean[:, np.newaxis, np.newaxis]
            self.data_list_u[i] = self.data_list_u[i] / self.intensities_std[:, np.newaxis, np.newaxis]

    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, index):
        sequence_id = self.seq_map[index]
        offset = index - self.seq_begin[index]
        selected_x = self.data_list_x[sequence_id][:, offset*self.dilation:(offset+self.time_steps+1)*self.dilation:self.dilation]
        selected_u = self.data_list_u[sequence_id][:, offset*self.dilation:(offset+self.time_steps+1)*self.dilation:self.dilation]

        if self.reverse:
            x0 = selected_x[:, -1]
            u = selected_u[:, -2::-1]
            x = selected_x[:, -2::-1]
        else:
            x0 = selected_x[:, 0]
            u = selected_u[:, 1:]
            x = selected_x[:, 1:]

        return x0.astype(np.float32), x.astype(np.float32), u.astype(np.float32)

    def __iter__(self):
        pass


def get_muscle_sequences(cfg, is_train=True, reverse=False):
    dataset = MuscleSequence(cfg.DATASET.ROOT,
                             time_steps=cfg.DATASET.TIME_STEPS,
                             dilation=cfg.DATASET.DILATION,
                             reverse=reverse)
    adjacent = dataset.generate_adjacent(cfg.MODEL.ST_KERNEL_G[0])
    # update the normalize value
    norm_param = {
        'DATASET.X_MEAN': dataset.muscle_mean.tolist(),
        'DATASET.X_STD': dataset.muscle_std.tolist(),
        'DATASET.U_MEAN': dataset.intensities_mean.tolist(),
        'DATASET.U_STD': dataset.intensities_std.tolist()
    }
    if is_train:
        data_loader = DataLoader(dataset,
                                 batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
                                 shuffle=True,
                                 pin_memory=cfg.PIN_MEMORY,
                                 num_workers=cfg.WORKERS)
    else:
        data_loader = DataLoader(dataset,
                                 batch_size=cfg.VAL.BATCH_SIZE_PER_GPU,
                                 shuffle=False,
                                 pin_memory=cfg.PIN_MEMORY,
                                 num_workers=cfg.WORKERS)
    return data_loader, adjacent.astype(np.float32), norm_param


if __name__ == '__main__':
    muscle_dataset = MuscleSequence('./data/larva/ventral.pkl', 48, 1, reverse=True)
    loader = DataLoader(muscle_dataset, 
                        batch_size=2,
                        shuffle=True,
                        num_workers=2)
    for step, (x0, x, u) in enumerate(loader):
        print(step, x0.shape, x.shape, u.shape)
    print(muscle_dataset.muscle_mean, muscle_dataset.muscle_std,
          muscle_dataset.intensities_mean, muscle_dataset.intensities_std) 
