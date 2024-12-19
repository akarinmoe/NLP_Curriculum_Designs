# Exp_Train.py

import torch
import torch.nn as nn
import time
import json
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from Exp_DataSet import Corpus  # 确保这个文件存在并正确实现
from Exp_Model import BiLSTM_model, Transformer_model, TransformerWithBertInit

def train_one_epoch(model, data_loader, loss_function, optimizer, device, batch_size):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    tqdm_iterator = tqdm(data_loader, dynamic_ncols=True, desc='Training')

    for data in tqdm_iterator:
        if isinstance(model, (TransformerWithBertInit)):
            # 数据集返回 (input_ids, labels)
            input_ids, labels = data

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # 检查 input_ids 的类型
            if input_ids.dtype != torch.long:
                print(f"Converting input_ids from {input_ids.dtype} to torch.long")
                input_ids = input_ids.long()
            
            # 检查 input_ids 中的最大值
            max_id = input_ids.max().item()
            vocab_size = model.embed.num_embeddings
            if max_id >= vocab_size:
                raise ValueError(f"Found input_ids with index {max_id} >= vocab_size {vocab_size}")

            # 计算 attention_mask，假设 [PAD] 的 token id 为 0
            attention_mask = (input_ids != 0).long()
            outputs = model(input_ids, attention_mask=attention_mask)
        else:
            # 原有模型，数据集返回 (input_ids, labels)
            input_ids, labels = data

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # 检查 input_ids 的类型
            if input_ids.dtype != torch.long:
                print(f"Converting input_ids from {input_ids.dtype} to torch.long")
                input_ids = input_ids.long()
            
            # 检查 input_ids 中的最大值
            max_id = input_ids.max().item()
            vocab_size = model.embed.num_embeddings
            if max_id >= vocab_size:
                raise ValueError(f"Found input_ids with index {max_id} >= vocab_size {vocab_size}")

            outputs = model(input_ids)

        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        tqdm_iterator.set_postfix(loss=avg_loss, acc=avg_acc)

    tqdm_iterator.close()
    return avg_loss, avg_acc

def validate(model, data_loader, loss_function, device, batch_size):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        tqdm_iterator = tqdm(data_loader, dynamic_ncols=True, desc='Validation')
        for data in tqdm_iterator:
            if isinstance(model, (TransformerWithBertInit)):
                # 数据集返回 (input_ids, labels)
                input_ids, labels = data
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                # 检查 input_ids 的类型
                if input_ids.dtype != torch.long:
                    print(f"Converting input_ids from {input_ids.dtype} to torch.long")
                    input_ids = input_ids.long()
                
                # 检查 input_ids 中的最大值
                max_id = input_ids.max().item()
                vocab_size = model.embed.num_embeddings
                if max_id >= vocab_size:
                    raise ValueError(f"Found input_ids with index {max_id} >= vocab_size {vocab_size}")

                # 计算 attention_mask，假设 [PAD] 的 token id 为 0
                attention_mask = (input_ids != 0).long()
                outputs = model(input_ids, attention_mask=attention_mask)
            else:
                # 原有模型，数据集返回 (input_ids, labels)
                input_ids, labels = data
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                # 检查 input_ids 的类型
                if input_ids.dtype != torch.long:
                    print(f"Converting input_ids from {input_ids.dtype} to torch.long")
                    input_ids = input_ids.long()
                
                # 检查 input_ids 中的最大值
                max_id = input_ids.max().item()
                vocab_size = model.embed.num_embeddings
                if max_id >= vocab_size:
                    raise ValueError(f"Found input_ids with index {max_id} >= vocab_size {vocab_size}")

                outputs = model(input_ids)

            loss = loss_function(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            tqdm_iterator.set_postfix(loss=avg_loss, acc=avg_acc)
        
        tqdm_iterator.close()
    return avg_loss, avg_acc

def predict(model, data_loader, dataset, device, output_folder):
    model.eval()
    test_ids = []
    test_pred = []

    with torch.no_grad():
        tqdm_iterator = tqdm(data_loader, dynamic_ncols=True, desc='Predicting')
        for data in tqdm_iterator:
            if isinstance(model, (TransformerWithBertInit)):
                # 数据集返回 (input_ids, ids)
                input_ids, ids = data
                input_ids = input_ids.to(device)

                # 检查 input_ids 的类型
                if input_ids.dtype != torch.long:
                    print(f"Converting input_ids from {input_ids.dtype} to torch.long")
                    input_ids = input_ids.long()
                
                # 检查 input_ids 中的最大值
                max_id = input_ids.max().item()
                vocab_size = model.embed.num_embeddings
                if max_id >= vocab_size:
                    raise ValueError(f"Found input_ids with index {max_id} >= vocab_size {vocab_size}")

                # 计算 attention_mask，假设 [PAD] 的 token id 为 0
                attention_mask = (input_ids != 0).long()
                outputs = model(input_ids, attention_mask=attention_mask)
            else:
                # 原有模型，数据集返回 (input_ids, ids)
                input_ids, ids = data
                input_ids = input_ids.to(device)

                # 检查 input_ids 的类型
                if input_ids.dtype != torch.long:
                    print(f"Converting input_ids from {input_ids.dtype} to torch.long")
                    input_ids = input_ids.long()
                
                # 检查 input_ids 中的最大值
                max_id = input_ids.max().item()
                vocab_size = model.embed.num_embeddings
                if max_id >= vocab_size:
                    raise ValueError(f"Found input_ids with index {max_id} >= vocab_size {vocab_size}")

                outputs = model(input_ids)
            
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            test_ids += ids.tolist()
            test_pred += preds
        tqdm_iterator.close()
    
    # 写入文件
    with open(os.path.join(output_folder, "predict.json"), "w", encoding='utf-8') as f:
        for idx, label_idx in enumerate(test_pred):
            one_data = {
                "id": test_ids[idx],
                "pred_label_desc": dataset.dictionary.idx2label[label_idx][1]
            }
            json_data = json.dumps(one_data, ensure_ascii=False)
            f.write(json_data + "\n")

def main():
    dataset_folder = '/home/xwc/nlp/tnews'
    output_folder = './output'
    os.makedirs(output_folder, exist_ok=True)

    # 环境检查
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 准备数据
    dataset = Corpus(dataset_folder, max_token_per_sent=50)
    
    # 设置 num_class 为标签数量
    num_class = len(dataset.dictionary.idx2label)
    print(f"Number of classes: {num_class}")
    
    # 超参数
    # 这里设置 d_emb 为与 BERT 相同的 768
    d_emb = 768  # 与 BERT 相同的嵌入维度
    d_hid = 3072
    nhead = 12
    nlayers = 6  # 可以根据需求调整，确保不超过BERT的层数
    dropout = 0.2
    batch_size = 16
    num_epochs = 5
    lr = 2e-5  # 对于预训练模型，通常使用较低的学习率

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)
    
    # 方式1: 使用预训练BERT参数初始化自定义Transformer
    model = TransformerWithBertInit(
        d_emb=d_emb,
        d_hid=d_hid,
        nhead=nhead,
        nlayers=nlayers,
        dropout=dropout,
        num_class=num_class,  # 使用实际类别数量
        pretrained_model_name='bert-base-chinese'
    )
    # Move to device after parameter copying
    model.to(device)
    
    # 方式2: 使用原始的Transformer_model（不使用预训练参数）
    # model = Transformer_model(
    #     vocab_size=dataset.tokenizer.vocab_size,
    #     ntoken=50,
    #     d_emb=300,
    #     d_hid=2048,
    #     nhead=6,
    #     nlayers=6,
    #     dropout=0.2,
    #     embedding_weight=dataset.embedding_weight
    # ).to(device)
    
    # 方式3: 使用BiLSTM_model
    # model = BiLSTM_model(
    #     vocab_size=dataset.tokenizer.vocab_size,
    #     ntoken=50,
    #     d_emb=300,
    #     d_hid=80,
    #     nlayers=1,
    #     dropout=0.2,
    #     embedding_weight=dataset.embedding_weight
    # ).to(device)
    
    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()
    
    # 设置优化器
    # 对于预训练模型，建议使用AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    max_valid_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, data_loader_train, loss_function, optimizer, device, batch_size)
        valid_loss, valid_acc = validate(model, data_loader_valid, loss_function, device, batch_size)
        
        if valid_acc > max_valid_acc:
            # 保存模型参数
            torch.save(model.state_dict(), os.path.join(output_folder, "model.ckpt"))
            max_valid_acc = valid_acc
        
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
              f"Valid Loss={valid_loss:.4f}, Valid Acc={valid_acc*100:.2f}%")
    
    # 加载最佳模型
    if isinstance(model, (TransformerWithBertInit)):
        model.load_state_dict(torch.load(os.path.join(output_folder, "model.ckpt")))
    else:
        model = Transformer_model(
            vocab_size=dataset.tokenizer.vocab_size,
            ntoken=50,
            d_emb=300,
            d_hid=2048,
            nhead=6,
            nlayers=6,
            dropout=0.2,
            embedding_weight=dataset.embedding_weight
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(output_folder, "model.ckpt")))
    
    # 对测试集进行预测
    predict(model, data_loader_test, dataset, device, output_folder)

if __name__ == '__main__':
    main()
