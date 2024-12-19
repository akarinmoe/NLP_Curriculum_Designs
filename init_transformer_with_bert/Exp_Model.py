# Exp_Model.py

import torch.nn as nn
import torch
import math
from transformers import BertModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 方式1: 保留原有Transformer架构，仅迁移BERT参数，并保持 d_emb=768
class TransformerWithBertInit(nn.Module):
    def __init__(self, d_emb=768, d_hid=3072, nhead=12, nlayers=6, dropout=0.2, num_class=15, pretrained_model_name='bert-base-chinese'):
        super(TransformerWithBertInit, self).__init__()
        # 加载预训练的BERT模型 (默认在CPU)
        bert = BertModel.from_pretrained(pretrained_model_name)
        
        # 初始化嵌入层，确保词汇表大小一致
        self.embed = nn.Embedding.from_pretrained(bert.embeddings.word_embeddings.weight, freeze=False)
        print(f"Embedding layer initialized with shape: {self.embed.weight.shape}")  # 应输出 [21128, 768]
        
        # 保持与BERT相同的嵌入维度，无需投影层
        # self.proj = nn.Linear(bert.config.hidden_size, d_emb)  # 不需要投影层，因为 d_emb=768
        
        # 保持原有的Transformer架构，设置 batch_first=True
        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            activation='gelu',  # 与BERT一致
            batch_first=True     # 设置 batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        self.dropout = nn.Dropout(dropout)
        
        # 分类器
        self.classifier = nn.Linear(d_emb, num_class)
        
        # 删除 BERT 模型以释放内存
        del bert
    
    def forward(self, input_ids, attention_mask=None):
        # 输入检查和调试
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids should be torch.long, but got {input_ids.dtype}")
        
        x = self.embed(input_ids)  # [batch_size, seq_len, 768]
        # x = self.proj(x)           # 不需要投影层
        x = self.pos_encoder(x)    # [batch_size, seq_len, 768]
        x = self.transformer_encoder(x)  # [batch_size, seq_len, 768]
        x = self.dropout(x)
        
        # Pooling策略
        mean_pooled = torch.mean(x, dim=1)  # [batch_size, 768]
        max_pooled, _ = torch.max(x, dim=1)  # [batch_size, 768]
        scores = torch.norm(x, p=2, dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        att_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        att_pooled = torch.sum(x * att_weights, dim=1)  # [batch_size, 768]
        combined = mean_pooled + max_pooled + att_pooled  # [batch_size, 768]
        logits = self.classifier(combined)  # [batch_size, num_class]
        return logits

# 方式2: 原始的Transformer_model（不使用预训练参数）
class Transformer_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=300, d_hid=2048, nhead=6, nlayers=6, dropout=0.2, embedding_weight=None):
        super(Transformer_model, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)
        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb, 
            nhead=nhead, 
            dim_feedforward=d_hid, 
            dropout=dropout, 
            activation='relu', 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_emb, 10)  # num_classes=15
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)  # [batch_size, seq_len, d_emb]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        # Pooling策略
        mean_pooled = torch.mean(x, dim=1)  # [batch_size, d_emb]
        max_pooled, _ = torch.max(x, dim=1)  # [batch_size, d_emb]
        scores = torch.norm(x, p=2, dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        att_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        att_pooled = torch.sum(x * att_weights, dim=1)  # [batch_size, d_emb]
        combined = mean_pooled + max_pooled + att_pooled  # [batch_size, d_emb]
        logits = self.classifier(combined)  # [batch_size, num_class]
        return logits

# 方式3: 使用BiLSTM_model
class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=300, d_hid=80, nlayers=1, dropout=0.2, embedding_weight=None):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)
        self.lstm = nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=nlayers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        num_class = 15
        self.classifier = nn.Linear(d_hid*2, num_class)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)  # [batch_size, seq_len, d_emb]
        x, _ = self.lstm(x)
        x = self.dropout(x)
        mean_pooled = torch.mean(x, dim=1)
        max_pooled, _ = torch.max(x, dim=1)
        scores = torch.norm(x, p=2, dim=-1, keepdim=True)
        att_weights = torch.softmax(scores, dim=1)
        att_pooled = torch.sum(x * att_weights, dim=1)
        combined = mean_pooled + max_pooled + att_pooled
        logits = self.classifier(combined)
        return logits
