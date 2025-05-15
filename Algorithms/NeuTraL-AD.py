import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from datetime import timedelta
from tqdm import tqdm
import os
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
 
 
##############################
# Utility Functions
##############################
 
def format_time(avg_time):
    avg_time = timedelta(seconds=avg_time)
    total_seconds = int(avg_time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{str(avg_time.microseconds)[:3]}"
 
def compute_pre_recall_f1(target, score):
    normal_ratio = (target == 0).sum() / len(target)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='binary')
    return precision, recall, f1
 
##############################
# Data Preparation
##############################
 
def train_test_split(csv_path):
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, -1].values
    samples = df.iloc[:, :-1].values
 
    inliers = samples[labels == 0]  # normal samples
    outliers = samples[labels == 1]  # anomalies
 
    num_split = len(inliers) // 2
    train_data = inliers[:num_split]
    train_label = np.zeros(num_split)
    test_data = np.concatenate([inliers[num_split:], outliers], 0)
    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:] = 1
    return train_data, train_label, test_data, test_label
 
class CustomDataset(Dataset):
    def __init__(self, samples, labels):
        self.labels = labels
        self.samples = samples
        self.dim_features = samples.shape[1]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        sample = self.samples[idx]
        data = [torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)]
        return data
 
##############################
# Model Components
##############################
 
class TabTransformNet(nn.Module):
    def __init__(self, x_dim, h_dim, num_layers):
        super(TabTransformNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers-1):
            net.append(nn.Linear(input_dim, h_dim, bias=False))
            net.append(nn.ReLU())
            input_dim = h_dim
        net.append(nn.Linear(input_dim, x_dim, bias=False))
        self.net = nn.Sequential(*net)
 
    def forward(self, x):
        return self.net(x)
 
class TabEncoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, bias, num_layers, batch_norm):
        super(TabEncoder, self).__init__()
        enc = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            enc.append(nn.Linear(input_dim, h_dim, bias=bias))
            if batch_norm:
                enc.append(nn.BatchNorm1d(h_dim, affine=bias))
            enc.append(nn.ReLU())
            input_dim = h_dim
        self.enc = nn.Sequential(*enc)
        self.fc = nn.Linear(input_dim, z_dim, bias=bias)
 
    def forward(self, x):
        z = self.enc(x)
        z = self.fc(z)
        return z
 
class TabNets():
    def _make_nets(self, x_dim, config):
        enc_nlayers = config['enc_nlayers']
        try:
            hdim = config['enc_hdim']
            zdim = config['latent_dim']
            trans_hdim = config['trans_hdim']
        except:
            if 32<=x_dim <= 300:
                zdim = 32
                hdim = 64
                trans_hdim = x_dim
            elif x_dim<32:
                zdim = 2 * x_dim
                hdim = 2 * x_dim
                trans_hdim = x_dim
            else:
                zdim = 64
                hdim = 256
                trans_hdim = x_dim
        trans_nlayers = config['trans_nlayers']
        num_trans = config['num_trans']
        batch_norm = config['batch_norm']
        enc = TabEncoder(x_dim, hdim, zdim, config['enc_bias'], enc_nlayers, batch_norm)
        trans = nn.ModuleList([TabTransformNet(x_dim, trans_hdim, trans_nlayers) for _ in range(num_trans)])
        return enc, trans
 
class TabNeutralAD(nn.Module):
    def __init__(self, model, x_dim, config):
        super(TabNeutralAD, self).__init__()
        self.enc, self.trans = model._make_nets(x_dim, config)
        self.num_trans = config['num_trans']
        self.trans_type = config['trans_type']
        self.device = config['device']
        try:
            self.z_dim = config['latent_dim']
        except:
            if 32<=x_dim <= 300:
                self.z_dim = 32
            elif x_dim<32:
                self.z_dim = 2 * x_dim
            else:
                self.z_dim = 64
 
    def forward(self, x):
        x = x.type(torch.FloatTensor).to(self.device)
        x_T = torch.empty(x.shape[0], self.num_trans, x.shape[-1]).to(x)
        for i in range(self.num_trans):
            mask = self.trans[i](x)
            if self.trans_type == 'forward':
                x_T[:, i] = mask
            elif self.trans_type == 'mul':
                mask = torch.sigmoid(mask)
                x_T[:, i] = mask * x
            elif self.trans_type == 'residual':
                x_T[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1), x_T], 1)
        zs = self.enc(x_cat.reshape(-1, x.shape[-1]))
        zs = zs.reshape(x.shape[0], self.num_trans+1, self.z_dim)
        return zs
 
class DCL(nn.Module):
    def __init__(self, temperature=0.1):
        super(DCL, self).__init__()
        self.temp = temperature
 
    def forward(self, z, eval=False):
        z = F.normalize(z, p=2, dim=-1)
        batch_size, num_trans, z_dim = z.shape
 
        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1)) / self.temp)
 
        # Create mask (no diagonal)
        mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        eye = torch.eye(num_trans, device=mask.device, dtype=torch.bool).unsqueeze(0)
        mask = mask.logical_and(~eye)
 
        sim_matrix_masked = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
 
        # Positive similarity: original sample vs its transformations
        z_ori = z[:, 0]  # [batch_size, z_dim]
        z_trans = z[:, 1:]  # [batch_size, num_trans-1, z_dim]
 
        pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), dim=-1) / self.temp)
 
        # Transformed samples similarities
        trans_matrix = sim_matrix_masked[:, 1:].sum(dim=-1)
 
        K = num_trans - 1
        scale = 1 / abs(K * np.log(1.0 / K))
 
        loss_tensor = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale
 
        if eval:
            score = loss_tensor.sum(1)
            return score
        else:
            loss = loss_tensor.sum(1)
            return loss
 
##############################
# Trainer
##############################
 
class NeutralAD_trainer:
    def __init__(self, model, loss_function, device='cuda'):
        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.model = model.to(self.device)
 
    def _train(self, train_loader, optimizer):
        self.model.train()
        loss_all = 0
        for data in train_loader:
            samples, _ = data
            samples = samples.to(self.device)
            z = self.model(samples)
            loss = self.loss_fun(z)
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()
            loss_all += loss.sum()
        return loss_all.item() / len(train_loader.dataset)
 
    def detect_outliers(self, loader, cls=1):
        model = self.model
        model.eval()
 
        loss_in = 0
        loss_out = 0
        target_all = []
        score_all = []
        loop = tqdm(loader, desc="Testing", leave=False)
        for data in loader:
            with torch.no_grad():
                try:
                    samples, labels = data
                except:
                    samples = data
                    labels = data.y!=cls
                z = model(samples)
                score = self.loss_fun(z, eval=True)
                loss_in += score[labels == 0].sum()
                loss_out += score[labels == 1].sum()
                target_all.append(labels.numpy())
                score_all.append(score.cpu().numpy())
        score_all = np.concatenate(score_all)
        target_all = np.concatenate(target_all)
        return loss_in.item() / (target_all == 0).sum(), loss_out.item() / (target_all == 1).sum(),target_all, score_all
 
##############################
# Main Code
##############################
 
urls = []
 
results = []
 
for url in tqdm(urls, desc="Datasets"):
    print(f"Processing: {url}")
    try:
        # Data Preparation
        train_data, train_label, test_data, test_label = train_test_split(url)
        train_dataset = CustomDataset(train_data, train_label)
        test_dataset = CustomDataset(test_data, test_label)
 
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
 
        # Model Config
        config = {
            'device': device,
            'latent_dim': 24,
            'enc_hdim': 24,
            'enc_nlayers': 5,
            'num_trans': 11,
            'trans_nlayers': 2,
            'trans_hdim': 24,
            'trans_type': 'residual',
            'enc_bias': False,
            'batch_norm': False
        }
 
        model = TabNeutralAD(TabNets(), x_dim=train_data.shape[1], config=config).to(device)
        loss_function = DCL(temperature=0.1)
        trainer = NeutralAD_trainer(model, loss_function, device=config['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
        for epoch in range(1, 101):
            loss = trainer._train(train_loader, optimizer)
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch [{epoch}/100] - Training Loss: {loss:.4f}")
 
        testin_loss,testout_loss,target, score = trainer.detect_outliers(test_loader)

        precision, recall, f1 = compute_pre_recall_f1(target, score)
        f2 = (5 * precision * recall) / (4 * precision + recall + 1e-8)
        auc = roc_auc_score(target, score)
 
        print(f"Inlier Loss: {testin_loss:.4f}")
        print(f"Outier Loss: {testout_loss:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F-1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"F-2 Score: {f2:.4f}")
        
        results.append({
            "Dataset URL": url,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "F2 Score": f2,
            "AUC-ROC": auc
        })
 
    except Exception as e:
        print(f"Failed on {url} with error: {str(e)}")
        continue
 
# Save to Excel
df_results = pd.DataFrame(results)
df_results.to_excel("results1.xlsx", index=False)
print("\n✅ Results saved to 'results1.xlsx' successfully!")
 