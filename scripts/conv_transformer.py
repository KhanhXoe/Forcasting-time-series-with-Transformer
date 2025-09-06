import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class PositionalInputEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalInputEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B, L, D] or [B, L, input_dim]
        return self.pe[:, :x.size(1), :].expand(x.size(0), -1, -1)  # [B, L, D]
    
class ValueEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, device):
        super(ValueEmbedding, self).__init__()
        
        self.fc = nn.Linear(input_dim, d_model, device=device)

    def forward(self, x):
        x = self.fc(x)
        return x

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()
        
        self.device = "cuda"
        
        self.hour_list = torch.tensor([1,4,7,10,13,16,19,22]).to(self.device)
        self.day_list = torch.tensor(np.arange(1, 32)).to(self.device)
        self.month_list = torch.tensor(np.arange(1, 13)).to(self.device)
        
        self.hour_embed = nn.Embedding(len(self.hour_list), d_model)
        self.day_embed = nn.Embedding(len(self.day_list), d_model)
        self.month_embed = nn.Embedding(len(self.month_list), d_model)

        init.xavier_uniform_(self.hour_embed.weight)
        init.xavier_uniform_(self.day_embed.weight)
        init.xavier_uniform_(self.month_embed.weight)
        
        self.hour_norm = nn.LayerNorm(d_model)
        self.day_norm = nn.LayerNorm(d_model)
        self.month_norm = nn.LayerNorm(d_model)

        self.temporal_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x_mark):
        
        x_hours = x_mark[:, :, 0].reshape(x_mark.shape[0], -1).to(self.device)
        x_days = x_mark[:, :, 1].reshape(x_mark.shape[0], -1).to(self.device)
        x_months = x_mark[:, :, 2].reshape(x_mark.shape[0], -1).to(self.device)

        hour_indices = (x_hours.unsqueeze(-1) == self.hour_list ).float().argmax(dim=-1).to(self.device)
        day_indices = (x_days.unsqueeze(-1) == self.day_list).float().argmax(dim=-1).to(self.device)
        month_indices = (x_months.unsqueeze(-1) == self.month_list).float().argmax(dim=-1).to(self.device)

        hour_x = self.hour_norm(self.hour_embed(hour_indices))
        day_x = self.day_norm(self.day_embed(day_indices))
        month_x = self.month_norm(self.month_embed(month_indices))
        
        out = torch.cat([day_x, month_x], dim=-1)  # shape: [B, T, 3*d_model]
        out = self.temporal_proj(out)     

        return out.to(self.device)
    
class DataEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, device, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.device = device
        
        self.value_embedding = ValueEmbedding(input_dim, d_model, device= device)
        self.position_embedding = PositionalInputEmbedding(d_model)
        self.temporal_embedding = TemporalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.prj = nn.Linear(
            in_features = d_model*3,
            out_features = d_model
        )
        self.prj2 = nn.Linear(
            in_features= d_model,
            out_features= d_model
        )
    
    def forward(self, x, x_times):
        value_emb = self.value_embedding(x)              
        pos_emb = self.position_embedding(x)             
        temp_emb = self.temporal_embedding(x_times)         
        x = torch.cat([value_emb, pos_emb, temp_emb], dim=-1)
        #x = value_emb + pos_emb + temp_emb
        x = self.prj(x)                                        
    
        return self.dropout(x).to(self.device)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        self.kernel_size = 7

        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size= self.kernel_size, padding=(self.kernel_size-1)//2, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size= self.kernel_size, padding=(self.kernel_size-1)//2, bias=False)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        q = self.conv_q(query.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.conv_k(key.permute(0, 2, 1)).permute(0, 2, 1)

        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        
        def reshape(x):
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        q, k, v = map(reshape, (q, k, v))
        self.scale = torch.tensor(1.0 / math.sqrt(self.d_k))
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1) 
        attn_probs = self.dropout(attn_probs)
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out_linear(output)
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.kernel_size = 3

        self.attn = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.omega = nn.Parameter(torch.ones(1, 1, d_model))
        nn.init.xavier_normal_(self.omega, gain=1.0)
        self.omega2 = nn.Parameter(torch.ones(1, 1, d_model))
        nn.init.xavier_normal_(self.omega2, gain=1.0)

    def forward(self, x, mask=None):
        shortcut = x * self.omega
        x = self.norm1(x)
        attn_output = self.attn(x, x, x, mask) 
        attn_output = self.dropout(attn_output) 
        attn_output = attn_output + shortcut

        shortcut2 = attn_output * self.omega2
        attn_output = self.norm1(attn_output)
        ffn_output = self.dropout(self.ffn(attn_output))
        ffn_output = ffn_output + shortcut2
        return self.norm2(ffn_output)
    
class Encoder(nn.Module):
    def __init__(
        self, pred_dim, 
        d_model, num_layers,
        num_heads, d_ff, 
        device, dropout=0.1
        ):
        super(Encoder, self).__init__()
        
        self.embedding = DataEmbedding(pred_dim, d_model, device, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model, device= device)

    def forward(self, x, x_times, mask=None):
        x = self.embedding(x, x_times)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout
        )
        
        self.cross_attn = MultiHeadAttention(
            d_model, num_heads, dropout
        )
        
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        x_norm = self.norm1(x)
        self_attn_output = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout(self_attn_output)

        x_norm = self.norm2(x)
        cross_attn_output = self.cross_attn(x_norm, enc_output, enc_output, src_mask)
        x = x + self.dropout(cross_attn_output)
        
        x_norm = self.norm3(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)

        return x
    

class Decoder(nn.Module):
    def __init__(self, tar_dim, d_model, num_layers, num_heads, d_ff, device, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = DataEmbedding(tar_dim, d_model, device, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model, device= device)

    def forward(self, x, x_times, enc_output, tgt_mask= None, src_mask= None):
        x = self.embedding(x, x_times)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self, pred_dim, tar_dim, device, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.device = device
        self.encoder = Encoder(pred_dim, d_model, num_layers, num_heads, d_ff, device, dropout)
        self.decoder = Decoder(tar_dim, d_model, num_layers, num_heads, d_ff, device, dropout)
        self.out = nn.Linear(d_model, tar_dim)

    def generate_mask(self, x_values, y_values):
        batch_size = x_values.size(0)
        pred_len = y_values.size(1)

        past_mask = None
        pred_mask = torch.triu(
            torch.ones((pred_len, pred_len), device=self.device), diagonal=1
        ).bool()  

        pred_mask = pred_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, tgt_seq_len, tgt_seq_len]
        pred_mask = pred_mask.expand(batch_size, 1, -1, -1)  # [batch_size, 1, tgt_seq_len, tgt_seq_len]

        return past_mask, pred_mask
    
    def forward(self, x_values, x_times, y_shifted, y_shifted_times, past_mask=None, pred_mask=None): 
        past_mask, pred_mask = self.generate_mask(x_values, y_shifted)
        enc_output = self.encoder(x_values, x_times, past_mask)
        dec_output = self.decoder(y_shifted, y_shifted_times, enc_output, pred_mask, past_mask) 
        return self.out(dec_output)

    def inference_reusing(self, x_values, x_times, y_label_times):

        self.eval()
        batch_size = x_values.size(0)
        pred_len = y_label_times.size(1)
        past_mask = None
        y_pred = []

        enc_output = self.encoder(x_values, x_times, past_mask)
        dec_input = x_values[:, -1:, :]
        dec_time_input = x_times[:, -1:, :]

        for i in range(pred_len):
            _, pred_mask = self.generate_mask(x_values, dec_input)
            dec_output = self.decoder(dec_input, dec_time_input, enc_output, pred_mask, past_mask)
            pred = self.out(dec_output)

            dec_input = torch.concat([dec_input, pred[:, -1:, :]], dim=1)
            dec_time_input = torch.concat([dec_time_input, y_label_times[:, i:i+1, :]], dim=1)

        return dec_input[:, 1:, :]