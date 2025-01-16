import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, Trainer, TrainingArguments, ViTModel
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
        self.encoder = ViTModel.from_pretrained(model_name).to(self.device)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, images):
        images = images.to(self.device)
        outputs = self.encoder(images)
        return outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_size]

# 特征对齐器
class FeatureAligner(nn.Module):
    def __init__(self, image_feature_dim, token_dim, device):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(image_feature_dim, token_dim).to(self.device)

    def forward(self, image_features):
        image_features = image_features.to(self.device)
        return self.linear(image_features)

# 多模态模型
class MultimodalModel(nn.Module):
    def __init__(self, image_encoder, llm, aligner, device_0, device_1):
        super().__init__()
        self.image_encoder = image_encoder.to(device_0)
        self.llm = llm.to(device_1)
        self.aligner = aligner.to(device_0)
        self.device_0 = device_0
        self.device_1 = device_1

    def forward(self, input_ids, attention_mask, image_A, image_B):
        image_A = image_A.to(self.device_0)
        image_B = image_B.to(self.device_0)
        input_ids = input_ids.to(self.device_1)
        attention_mask = attention_mask.to(self.device_1)

        # 图像编码
        features_A = self.image_encoder(image_A)
        features_B = self.image_encoder(image_B)

        print("features_A: ", features_A)
        print("features_B: ", features_B)

        # 对齐特征
        aligned_features_A = self.aligner(features_A)
        aligned_features_B = self.aligner(features_B)

        # 合并对齐后的图像特征并转移到 LLM 模型的设备
        multimodal_inputs = torch.cat([aligned_features_A, aligned_features_B], dim=1).to(self.device_1)

        # 获取文本嵌入
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # 拼接图像特征和文本嵌入
        combined_inputs = torch.cat([multimodal_inputs, inputs_embeds], dim=1)

        # 调用 LLM 进行推理
        outputs = self.llm(attention_mask=attention_mask, inputs_embeds=combined_inputs)
        return outputs

# 数据集定义
class ChangeDetectionDataset(Dataset):
    def __init__(self, data, root_dir, processor, tokenizer, max_length=128):
        self.data = data
        self.root_dir = root_dir
        self.processor = processor
        self.tokenizer = tokenizer
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

        # image_A_processed = image_A_processed.to("cuda:0")
        # image_B_processed = image_B_processed.to("cuda:0")

        print("image_A_processed: ", image_A_processed)
        print("image_B_processed: ", image_B_processed)

        # 创建对话式指令模板：Describe the difference between these two images.
        instruction = "Describe the difference between these two images."

        # 选择一句描述，作为模型的输出
        output_sentence = random.choice(sample['sentences'])['raw']
        labels = self.tokenizer(output_sentence, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        # 将 [IMG] 和 [/IMG] token 加入图像输入（而不是传递图像 Tensor）
        image_A_tokenized = "[IMG] " + str(image_A_processed.shape) + " [/IMG]"
        image_B_tokenized = "[IMG] " + str(image_B_processed.shape) + " [/IMG]"

        print(image_A_tokenized)
        print(image_B_tokenized)

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
        encoding = self.tokenizer(q_text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        # 提取文本输入和注意力掩码
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,  # 训练时，labels 为目标生成的文本
            'image_A': image_A_processed,  # 传递图像A
            'image_B': image_B_processed   # 传递图像B
        }

# 加载数据
def load_data(captions_file, split):
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)['images']
    return [item for item in captions_data if item['split'] == split]

# 训练函数
def train():
    data_dir = "./Levir-CC-dataset/images"
    captions_file = "./Levir-CC-dataset/LevirCCcaptions.json"
    train_data = load_data(captions_file, "train")
    val_data = load_data(captions_file, "val")

    train_dataset = ChangeDetectionDataset(train_data, data_dir, processor=PROCESSOR, tokenizer=TOKENIZER)
    val_dataset = ChangeDetectionDataset(val_data, data_dir, processor=PROCESSOR, tokenizer=TOKENIZER)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device_0 = torch.device("cuda:0")
    device_1 = torch.device("cuda:1")

    image_encoder = ImageEncoder(IMAGE_ENCODER_MODEL, device_0)
    llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL).to("cuda:0")
    aligner = FeatureAligner(image_feature_dim=image_encoder.encoder.config.projection_dim, 
                             token_dim=llm.config.hidden_size, device=device_0)

    model = MultimodalModel(image_encoder, llm, aligner, device_0, device_1)

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        logging_dir='./logs',
        evaluation_strategy="epoch",  # 设置为每个 epoch 后评估一次
        save_steps=1000,
        logging_steps=100,
        # device="cuda:0"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # 这里传递 eval_dataset
        tokenizer=TOKENIZER,
        data_collator=None,  # 使用默认的数据拼接器
    )

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model(SAVE_DIR)
    print(f"Model saved at {SAVE_DIR}")

if __name__ == "__main__":
    train()
