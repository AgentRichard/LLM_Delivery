from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                          AutoConfig, TrainingArguments, 
                          Trainer, BitsAndBytesConfig, DataCollatorForSeq2Seq)
import torch
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

### load 模型与数据集地址
model_path = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/_cache"
dataset_path = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/datas/ruozhiba"
model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model_config.cache_max_batch_size = None
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

### 量化配置
_bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,                   #模型的权重将被量化为4位整数，从而减少内存消耗并加快计算速度
    bnb_4bit_use_double_quant = True,      #模型的权重将被量化两次，以保持更高的精度。双重量化可以在保持较低位宽的同时减少量化误差，从而提高模型的性能。
    bnb_4bit_quant_type = "nf4",           #即“Near Forward Four-bit”。这是一种特定的4位量化方法，旨在保持模型的前向传递性能接近于全精度模型的性能。NF4量化通常用于确保在量化后模型的预测质量不会显著下降。
    bnb_4bit_compute_dtype = torch.float32 #尽管权重被量化为4位，但在实际的计算过程中仍然使用较高的精度来保持准确性。这是因为即使权重被量化，使用较高的精度进行中间计算可以帮助减少累积误差。
)

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             config = model_config,
                                             quantization_config = _bnb_config,
                                             trust_remote_code=True)
### 自己写的模型暂时还没办法使用量化? 因为没有预留API接口

### 加载处理数据
dataset = load_dataset(dataset_path, split="train")

def dataset_preprocess(data):
    MAX_LENGTH = 256
    prompt_ids = tokenizer(f"<s>user\n{data["query"]}</s>\n<s>assistant\n")
    response_ids = tokenizer(data["response"] + tokenizer.eos_token)
    input_ids = prompt_ids["input_ids"] + response_ids["input_ids"]
    attention_mask = prompt_ids["attention_mask"] + response_ids["attention_mask"]
    labels = [-100] * len(prompt_ids["input_ids"]) + response_ids["input_ids"]
    
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]
    return {
        "input_ids" : input_ids,
        "attention_mask" : attention_mask,
        "labels" : labels
    }
    
dataset = dataset.map(dataset_preprocess, remove_columns=dataset.column_names)
dataset.shuffle()

### 模型适配lora配置
lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    r = 100,
    target_modules = "all-linear",
    lora_alpha = 16
)
model = get_peft_model(model=model, peft_config=lora_config)
model.print_trainable_parameters()

training_argument = TrainingArguments(
    output_dir = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/checkpoint_qlora",
    per_device_train_batch_size = 5,
    gradient_accumulation_steps = 2,
    logging_steps = 10,
    save_steps = 500,
    num_train_epochs = 1
)

trainer = Trainer(
    model = model,
    args = training_argument,
    train_dataset = dataset,
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()

