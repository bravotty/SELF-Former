# -*- coding:utf-8 _*-
__author__ = 'xindiawei2'
__date__ = '6/9/2023 4:10 pm'
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
import math
from anndata import AnnData
import scanpy as sc


class EarlyStopper:
    def __init__(self, patience=3, min_delta=10):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def only_max(matrix):
    max_indices = np.amax(matrix, axis=1)
    result_matrix = np.where(matrix == max_indices[:, np.newaxis], matrix, 0)
    return  result_matrix



def normalize_one(x1, type='per_cell'):
    if type=='per_cell':
        x1 = normalize(x1, axis=1)
    elif type=='per_gene':
        x1 = normalize(x1, axis=0)
    elif type=='all':
        train_scaler = MinMaxScaler(feature_range=(0, 1))
        x1 = train_scaler.fit_transform(x1)
    elif type=='all_norm':
        train_scaler = StandardScaler()
        x1 = train_scaler.fit_transform(x1)
    elif type=='all_genes':
        x1_ann = AnnData(x1)
        sc.pp.normalize_total(x1_ann, target_sum=5, inplace=False)
        sc.pp.log1p(x1_ann)
        sc.pp.scale(x1_ann, zero_center=False, max_value=5)
        x1 = x1_ann.X
    return x1

def normalize_type(x1, x2, type='per_cell'):
    if type=='per_cell':
        x1 = normalize(x1, axis=1)
        x2 = normalize(x2, axis=1)
    elif type=='per_gene':
        x1 = normalize(x1, axis=0)
        x2 = normalize(x2, axis=0)
    elif type=='all':
        train_scaler = MinMaxScaler(feature_range=(0, 1))
        x1 = train_scaler.fit_transform(x1)

        test_scaler = MinMaxScaler(feature_range=(0, 1))
        x2 = test_scaler.fit_transform(x2)

    elif type=='all_norm':
        train_scaler = StandardScaler()
        x1 = train_scaler.fit_transform(x1)

        test_scaler = StandardScaler()
        x2 = test_scaler.fit_transform(x2)

    elif type=='10x':
        hvg = 1000
        x1_ann = AnnData(x1)
        sc.pp.normalize_total(x1_ann, target_sum=1e4)
        sc.pp.log1p(x1_ann)
        x1 = x1_ann.X

        x2_ann = AnnData(x2)
        sc.pp.normalize_total(x2_ann, target_sum=1e4)
        sc.pp.log1p(x2_ann)
        x2 = x2_ann.X
        return x1, x2, x1_ann, x2_ann

    elif type=='all_genes':
        x1_ann = AnnData(x1)
        sc.pp.normalize_total(x1_ann, target_sum=1e4)
        sc.pp.log1p(x1_ann)
        x1 = x1_ann.X
        x2_ann = AnnData(x2)
        sc.pp.normalize_total(x2_ann, target_sum=1e4)
        sc.pp.log1p(x2_ann)
        x2 = x2_ann.X
    return x1, x2


