import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import CrossEntropyLoss
from Exp_DataSet import Corpus
from Exp_Model import TransformerWithBertInit

def check_input_ids(input_ids, model):
    if input_ids.dtype != torch.long:
        print(f"Converting input_ids from {input_ids.dtype} to torch.long")
        input_ids = input_ids.long()

    max_id = input_ids.max().item()
    vocab_size = model.embed.num_embeddings
    if max_id >= vocab_size:
        raise ValueError(f"Found input_ids with index {max_id} >= vocab_size {vocab_size}")
    
    return input_ids

def train(model, data_loader, optimizer, criterion, device):
    model.train()  # Set model to training mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for data in tqdm(data_loader, dynamic_ncols=True, desc='Training'):
        input_ids, labels = data
        input_ids, labels = input_ids.to(device), labels.to(device)

        # Check input_ids for validity
        input_ids = check_input_ids(input_ids, model)

        # Attention mask (assuming PAD tokens are 0)
        attention_mask = (input_ids != 0).long()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # Compute loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data in tqdm(data_loader, dynamic_ncols=True, desc='Evaluating'):
            input_ids, labels = data
            input_ids, labels = input_ids.to(device), labels.to(device)

            input_ids = check_input_ids(input_ids, model)
            attention_mask = (input_ids != 0).long()

            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def predict(model, data_loader, dataset, device, output_folder):
    model.eval()
    test_ids = []
    test_pred = []

    with torch.no_grad():
        tqdm_iterator = tqdm(data_loader, dynamic_ncols=True, desc='Predicting')
        for data in tqdm_iterator:
            input_ids, ids = data
            input_ids = input_ids.to(device)

            input_ids = check_input_ids(input_ids, model)

            attention_mask = (input_ids != 0).long()
            outputs = model(input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            test_ids += ids.tolist()
            test_pred += preds
        tqdm_iterator.close()

    with open(os.path.join(output_folder, "predict.json"), "w", encoding='utf-8') as f:
        for idx, label_idx in enumerate(test_pred):
            one_data = {
                "id": test_ids[idx],
                "pred_label_desc": dataset.dictionary.idx2label[label_idx][1]
            }
            json_data = json.dumps(one_data, ensure_ascii=False)
            f.write(json_data + "\n")

def main():
    dataset_folder = '/home/xwc/nlp/tnews'  # 数据集文件夹路径
    output_folder = './output'  # 输出文件夹路径
    os.makedirs(output_folder, exist_ok=True)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    device = torch.device("cpu")
    print(f"Using device: {device}")

    dataset = Corpus(dataset_folder, max_token_per_sent=50)
    
    num_class = len(dataset.dictionary.idx2label)
    print(f"Number of classes: {num_class}")
    
    # 超参数
    d_emb = 768  # 与 BERT 相同的嵌入维度
    d_hid = 3072
    nhead = 12
    nlayers = 6
    dropout = 0.2
    batch_size = 16
    epochs = 1

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)

    model = TransformerWithBertInit(
        d_emb=d_emb,
        d_hid=d_hid,
        nhead=nhead,
        nlayers=nlayers,
        dropout=dropout,
        num_class=num_class,
        pretrained_model_name='bert-base-chinese'
    )

    model.to(device)
    
    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = CrossEntropyLoss()

    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train(model, data_loader_train, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, data_loader_valid, device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_folder, "model.ckpt"))

    # Test the model after training
    model.load_state_dict(torch.load(os.path.join(output_folder, "model.ckpt")))
    predict(model, data_loader_test, dataset, device, output_folder)

if __name__ == '__main__':
    main()
