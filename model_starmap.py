import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SELF_Former(nn.Module):
    def __init__(self, config):
        super(SELF_Former, self).__init__()
        tmp_dims_in = config['num_sample2']
        tmp_dims_in2 = config['num_sample1']
        in_dims_1 = 2048
        in_dims_2 = 1024
        in_3 = 1024
        self.linear_e1 = nn.Linear(tmp_dims_in, in_dims_1)
        self.linear_e2 = nn.Linear(in_dims_1, in_dims_2)

        self.linear_l1 = nn.Linear(in_dims_2, in_3)
        self.linear_l2 = nn.Linear(in_3*3, in_3)
        self.linear_l3 = nn.Linear(in_3, tmp_dims_in2)

        self.relu = nn.LeakyReLU(0.2)
        self.dp = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(in_dims_1)
        self.bn2 = nn.BatchNorm1d(in_dims_2)
        self.bn3 = nn.BatchNorm1d(in_3)

        self.ln1 = nn.LayerNorm(in_dims_1)
        self.ln2 = nn.LayerNorm(in_dims_2)
        self.ln3 = nn.LayerNorm(in_3)
        self.bn4 = nn.BatchNorm1d(tmp_dims_in2)
        self.sigmoid = nn.Sigmoid()

        self.linear_self_Q1 = nn.Linear(in_3, in_3)
        self.linear_self_K1 = nn.Linear(in_3, in_3)
        self.linear_self_V1 = nn.Linear(in_3, in_3)

        self.linear_self_Q2 = nn.Linear(in_dims_2, in_dims_2)
        self.linear_self_K2 = nn.Linear(in_dims_2, in_dims_2)
        self.linear_self_V2 = nn.Linear(in_dims_2, in_dims_2)

        self.linear_self_Q3 = nn.Linear(in_dims_2, in_dims_2)
        self.linear_self_K3 = nn.Linear(in_dims_2, in_dims_2)
        self.linear_self_V3 = nn.Linear(in_dims_2, in_dims_2)

        self.linear_self_Q4 = nn.Linear(in_dims_2, in_dims_2)
        self.linear_self_K4 = nn.Linear(in_dims_2, in_dims_2)
        self.linear_self_V4 = nn.Linear(in_dims_2, in_dims_2)

    def self_trans(self, lat):
        q = self.relu((self.linear_self_Q1(lat)))
        k = self.relu((self.linear_self_K1(lat)))
        v = self.relu((self.linear_self_V1(lat)))
        W_ = torch.matmul(q, k.T)  / math.sqrt(k.shape[1])
        W_ = F.softmax(W_, dim=1)
        diagonal_weights = torch.diag(W_)
        sorted_indices = torch.argsort(diagonal_weights, descending=True)
        selected_indices = sorted_indices[:len(sorted_indices) // 2]
        new_W_ = W_[selected_indices]
        new_lat = lat[selected_indices]
        new_W_ = new_W_.matmul(v)
        out_select = new_W_ + new_lat
        W_ = W_.matmul(v)
        out = W_
        q_s = self.relu((self.linear_self_Q4(out)))
        k_s = self.relu((self.linear_self_K4(out_select)))
        v_s = self.relu((self.linear_self_V4(out_select)))
        W_s = torch.matmul(q_s, k_s.T)  / math.sqrt(k.shape[1])
        W_s = F.softmax(W_s, dim=1)
        W_s = W_s.matmul(v_s)
        out_s = W_s
        return out_s

    def self_trans2(self, lat):
        q = self.relu((self.linear_self_Q2(lat)))
        k = self.relu((self.linear_self_K2(lat)))
        v = self.relu((self.linear_self_V2(lat)))
        W_ = torch.matmul(q, k.T) / math.sqrt(k.shape[1])
        W_ = F.softmax(W_, dim=1)
        W_ = W_.matmul(v)
        return W_

    def self_trans3(self, lat):
        q = self.relu((self.linear_self_Q3(lat)))
        k = self.relu((self.linear_self_K3(lat)))
        v = self.relu((self.linear_self_V3(lat)))
        W_ = torch.matmul(q, k.T) / math.sqrt(k.shape[1])
        W_ = F.softmax(W_, dim=1)
        W_ = W_.matmul(v)
        return W_

    def forward(self, x):
        latent = (self.relu((self.linear_e1(x.T))))
        latent = (self.relu((self.linear_e2(latent))))
        latent_trans1 = self.self_trans2(latent)
        latent_trans2 = self.self_trans3(latent)
        latent_st = (self.relu((self.linear_l1(latent))))
        latent_trans = self.self_trans(latent_st)
        latent_trans_input = torch.cat([latent_trans, latent_trans1, latent_trans2], dim=-1)
        latent_st2 = (self.relu(self.linear_l2(latent_trans_input)))
        xst = ((self.linear_l3(latent_st2)))
        out = xst.T
        return out

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.model_trans = SELF_Former(config)
    def forward(self, x_rna):
        x1_hat = self.model_trans(x_rna)
        return x1_hat









