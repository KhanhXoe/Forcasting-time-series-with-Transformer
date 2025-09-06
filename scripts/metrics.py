import torch
import numpy as np

def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

def nse(y_true, y_pred):
    """
    Nash–Sutcliffe efficiency (thường dùng trong thủy văn)
    """
    y_true_mean = torch.mean(y_true)
    num = torch.sum((y_true - y_pred) ** 2)
    den = torch.sum((y_true - y_true_mean) ** 2)
    return 1 - num / den if den > 0 else float("nan")