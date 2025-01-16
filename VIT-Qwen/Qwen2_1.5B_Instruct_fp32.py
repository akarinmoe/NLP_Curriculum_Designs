import torch
from torch import nn
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, Trainer, TrainingArguments, ViTModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import random

# 配置部分
IMAGE_ENCODER_MODEL = "/root/autodl-tmp/model/openai/clip-vit-large-patch14"
LLM_MODEL = "/root/autodl-tmp/model/Qwen/Qwen2-1.5B-Instruct"
TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL)
PROCESSOR = AutoProcessor.from_pretrained(IMAGE_ENCODER_MODEL)
SAVE_DIR = "/root/autodl-tmp/nlpcheckpoint"

# 图像编码器（ViT）定义
class ImageEncoder(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        self.encoder = ViTModel.from_pretrained(model_name).to(self.device)  # 确保模型在同一设备上
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, images):
        images = images.to(self.device)  # 确保输入图像在正确的设备上
        outputs = self.encoder(images)
        return outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_size]

# 特征对齐器
class FeatureAligner(nn.Module):
    def __init__(self, image_feature_dim, token_dim, device):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(image_feature_dim, token_dim).to(self.device)  # 确保线性层在正确设备

    def forward(self, image_features):
        image_features = image_features.to(self.device)  # 确保输入在正确设备
        return self.linear(image_features)

# 多模态模型
class MultimodalModel(nn.Module):
    def __init__(self, image_encoder, llm, aligner, device):
        super().__init__()
        self.image_encoder = image_encoder.to(device)
        self.llm = llm.to(device)
        self.aligner = aligner.to(device)
        self.device = device

        # 用于调整图像特征长度的池化层
        self.image_pooling = nn.AdaptiveAvgPool1d(128).to(self.device)  # 将图像特征长度调整为128

    def forward(self, input_ids, attention_mask, image_A, image_B, labels=None):
        # 图像输入转移到同一设备
        image_A = image_A.to(self.device)
        image_B = image_B.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # 图像编码
        features_A = self.image_encoder(image_A)
        features_B = self.image_encoder(image_B)

        # 对齐特征
        aligned_features_A = self.aligner(features_A)
        aligned_features_B = self.aligner(features_B)

        # 调整图像特征的序列长度为128
        aligned_features_A = aligned_features_A.permute(0, 2, 1)  # [batch_size, feature_dim, seq_len]
        aligned_features_B = aligned_features_B.permute(0, 2, 1)  # [batch_size, feature_dim, seq_len]

        # 使用池化层调整序列长度
        pooled_A = self.image_pooling(aligned_features_A)  # [batch_size, feature_dim, 128]
        pooled_B = self.image_pooling(aligned_features_B)  # [batch_size, feature_dim, 128]

        # 再将池化后的图像特征恢复维度
        pooled_A = pooled_A.permute(0, 2, 1)  # [batch_size, 128, feature_dim]
        pooled_B = pooled_B.permute(0, 2, 1)  # [batch_size, 128, feature_dim]

        # 合并图像特征
        multimodal_inputs = torch.cat([pooled_A, pooled_B], dim=1)  # [batch_size, 128, feature_dim * 2]

        # 强制将图像特征的长度调整为 128，并确保维度为 [1, 128, 1536]
        if multimodal_inputs.shape[1] > 128:
            multimodal_inputs = multimodal_inputs[:, :128, :]  # 截断图像特征到128
        elif multimodal_inputs.shape[1] < 128:
            # 如果图像特征长度小于128，填充到128
            padding_size = 128 - multimodal_inputs.shape[1]
            multimodal_inputs = torch.cat([multimodal_inputs, multimodal_inputs.new_zeros(1, padding_size, multimodal_inputs.shape[2])], dim=1)

        # 强制让 multimodal_inputs 的维度为 [1, 128, 1536]
        if multimodal_inputs.shape[1] != 128:
            multimodal_inputs = multimodal_inputs[:, :128, :]  # 如果维度大于128，截断
            # 如果维度小于128，填充到128
            padding_size = 128 - multimodal_inputs.shape[1]
            multimodal_inputs = torch.cat([multimodal_inputs, multimodal_inputs.new_zeros(1, padding_size, multimodal_inputs.shape[2])], dim=1)

        # 获取文本嵌入
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # 如果文本嵌入的长度超过128，需要做池化处理
        if inputs_embeds.shape[1] > 128:
            inputs_embeds = inputs_embeds[:, :128, :]  # 截断到 128 长度
        elif inputs_embeds.shape[1] < 128:
            # 如果文本嵌入的长度小于128，填充到128
            padding_size = 128 - inputs_embeds.shape[1]
            inputs_embeds = torch.cat([inputs_embeds, inputs_embeds.new_zeros(1, padding_size, inputs_embeds.shape[2])], dim=1)

        # 拼接图像特征和文本嵌入
        combined_inputs = torch.cat([multimodal_inputs, inputs_embeds], dim=1)

        # 强制调整 combined_inputs 的形状为 [1, 128, 1536]
        if combined_inputs.shape[1] > 128:
            combined_inputs = combined_inputs[:, :128, :]  # 截断到 128 长度
        elif combined_inputs.shape[1] < 128:
            # 如果 combined_inputs 的序列长度小于128，填充到128
            padding_size = 128 - combined_inputs.shape[1]
            combined_inputs = torch.cat([combined_inputs, combined_inputs.new_zeros(1, padding_size, combined_inputs.shape[2])], dim=1)

        # 打印检查 combined_inputs 的形状
        print(combined_inputs.shape)  # 应该是 [1, 128, 1536]

        # 调用 LLM 进行推理
        outputs = self.llm(attention_mask=attention_mask, inputs_embeds=combined_inputs, labels=labels)

        # 强行将 logits 和 labels 的 batch size 对齐
        if labels is not None:
            logits = outputs.logits
            batch_size = labels.size(0)

            # 如果 logits 的 batch_size 比 labels 大, 进行裁剪
            if logits.size(0) > batch_size:
                logits = logits[:batch_size]
            # 如果 logits 的 batch_size 比 labels 小, 进行填充
            elif logits.size(0) < batch_size:
                padding_size = batch_size - logits.size(0)
                logits = torch.cat([logits, logits.new_zeros(padding_size, logits.size(1), logits.size(2))], dim=0)

            # 返回修正后的 logits 和损失
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, outputs

        return outputs


# 数据集定义
class ChangeDetectionDataset(Dataset):
    def __init__(self, data, root_dir, processor, tokenizer, device, max_length=128):
        self.data = data
        self.root_dir = root_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        # 为了处理 [IMG] 和 [/IMG] token, 添加它们到 tokenizer 的特殊 token 列表
        special_tokens = {'additional_special_tokens': ['[IMG]', '[/IMG]']}
        self.tokenizer.add_special_tokens(special_tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_A_path = os.path.join(self.root_dir, sample['filepath'], "A", sample['filename'])
        image_B_path = os.path.join(self.root_dir, sample['filepath'], "B", sample['filename'])

        image_A = Image.open(image_A_path).convert("RGB")
        image_B = Image.open(image_B_path).convert("RGB")
        image_A_processed = self.processor(images=image_A, return_tensors="pt")["pixel_values"].squeeze(0)
        image_B_processed = self.processor(images=image_B, return_tensors="pt")["pixel_values"].squeeze(0)

        # 创建对话式指令模板：Describe the difference between these two images.
        instruction = "Describe the difference between these two images."

        # 选择一句描述，作为模型的输出
        output_sentence = random.choice(sample['sentences'])['raw']
        
        # 使用 tokenizer 处理 labels，同时启用 padding 和 truncation
        labels = self.tokenizer(output_sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        
        # 将 [IMG] 和 [/IMG] token 加入图像输入（而不是传递图像 Tensor）
        image_A_tokenized = "[IMG] " + str(image_A_processed.shape) + " [/IMG]"
        image_B_tokenized = "[IMG] " + str(image_B_processed.shape) + " [/IMG]"

        # 构建对话输入
        input_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
            {"role": "user", "content": "Image A and Image B are provided."},
            {"role": "user", "content": image_A_tokenized},  # 仅传递 [IMG] 标记
            {"role": "user", "content": image_B_tokenized},  # 仅传递 [IMG] 标记
            {"role": "assistant", "content": output_sentence}
        ]

        # 使用 Qwen 模型的对话模板
        q_text = self.tokenizer.apply_chat_template(input_conversation, tokenize=False, add_generation_prompt=True)

        # 注意：此时 q_text 是一个字符串，所以你需要先将其转换为 token IDs
        encoding = self.tokenizer(q_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

        # 提取文本输入和注意力掩码
        input_ids = encoding['input_ids'].squeeze(0).to(self.device)  # 确保在相同设备上
        attention_mask = encoding['attention_mask'].squeeze(0).to(self.device)  # 确保在相同设备上

        # 返回数据
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels['input_ids'].squeeze(0).to(self.device),  # 处理 labels 为 tensor 形式
            'image_A': image_A_processed.to(self.device),  # 确保图像A在相同设备上
            'image_B': image_B_processed.to(self.device)   # 确保图像B在相同设备上
        }

# 加载数据
def load_data(captions_file, split):
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)['images']
    return [item for item in captions_data if item['split'] == split]

# 训练函数
def train():

    device = torch.device("cpu")  # 强制使用 CPU

    data_dir = "./Levir-CC-dataset/images"
    captions_file = "./Levir-CC-dataset/LevirCCcaptions.json"
    train_data = load_data(captions_file, "train")
    val_data = load_data(captions_file, "val")

    train_dataset = ChangeDetectionDataset(train_data, data_dir, processor=PROCESSOR, tokenizer=TOKENIZER, device=device)
    val_dataset = ChangeDetectionDataset(val_data, data_dir, processor=PROCESSOR, tokenizer=TOKENIZER, device=device)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    image_encoder = ImageEncoder(IMAGE_ENCODER_MODEL, device)
    llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL).to(device)
    aligner = FeatureAligner(image_feature_dim=image_encoder.encoder.config.projection_dim, 
                             token_dim=llm.config.hidden_size, device=device)

    model = MultimodalModel(image_encoder, llm, aligner, device)

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        logging_dir='./logs',
        evaluation_strategy="epoch",  # 设置为每个 epoch 后评估一次
        save_strategy="epoch",  # 保存策略也设置为每个 epoch 后保存
        save_steps=1000,  # 如果使用 steps 则按步骤保存，但这里不需要
        logging_steps=100,
        load_best_model_at_end=True,  # 保证加载最优模型
        no_cuda=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # 这里传递 eval_dataset
        tokenizer=TOKENIZER,
        data_collator=None,  # 使用默认的数据拼接器
        compute_metrics=None,  # 如果你需要计算额外的评价指标，可以在这里传入
    )

    # 开始训练
    trainer.train()

# 训练过程
if __name__ == "__main__":
    train()
