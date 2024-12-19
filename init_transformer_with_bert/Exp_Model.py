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

class TransformerWithBertInit(nn.Module):
    def __init__(self, d_emb=768, d_hid=3072, nhead=12, nlayers=6, dropout=0.2, num_class=15, pretrained_model_name='bert-base-chinese'):
        super(TransformerWithBertInit, self).__init__()
        
        # 加载预训练的BERT模型
        bert = BertModel.from_pretrained(pretrained_model_name)
        
        # 初始化嵌入层，确保词汇表大小一致
        self.embed = nn.Embedding.from_pretrained(bert.embeddings.word_embeddings.weight, freeze=False)
        
        # 初始化 Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=512)
        
        # 创建 Transformer Encoder Layer 并从 BERT 提取权重
        encoder_layers = []
        for i in range(nlayers):
            attention_layer = bert.encoder.layer[i].attention
            feedforward_layer = bert.encoder.layer[i].output
            
            # 创建新的 TransformerEncoderLayer，并加载BERT的权重
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_emb,
                nhead=nhead,
                dim_feedforward=d_hid,  # 确保前馈层的维度合理，通常为d_emb或d_hid
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )

            # 从BERT中提取 query, key, value 的权重
            query_weight = attention_layer.self.query.weight
            key_weight = attention_layer.self.key.weight
            value_weight = attention_layer.self.value.weight
            query_bias = attention_layer.self.query.bias
            key_bias = attention_layer.self.key.bias
            value_bias = attention_layer.self.value.bias
            
            # 初始化 self-attention 的权重
            encoder_layer.self_attn.in_proj_weight.data = torch.cat([
                query_weight, key_weight, value_weight
            ], dim=0)  # 拼接 query, key, value 权重
            encoder_layer.self_attn.in_proj_bias.data = torch.cat([
                query_bias, key_bias, value_bias
            ], dim=0)  # 拼接 query, key, value 偏置
            
            # # 从BERT中提取前馈网络的权重
            # encoder_layer.linear1.weight.data = feedforward_layer.dense.weight
            # encoder_layer.linear1.bias.data = feedforward_layer.dense.bias
            
            # 将初始化好的encoder_layer添加到列表中
            encoder_layers.append(encoder_layer)
        
        # 使用 nn.ModuleList 来保存所有层
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        
        # Dropout层和分类器
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_emb, num_class)
        
        # 删除 BERT 模型以释放内存
        del bert
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)  # [batch_size, seq_len, 768]
        x = self.pos_encoder(x)    # [batch_size, seq_len, 768]
        
        # 按顺序通过每一层
        for layer in self.transformer_encoder:
            x = layer(x)  # [batch_size, seq_len, 768]
        
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
