from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          AutoConfig, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

### load 模型与数据集地址
model_path = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/_cache"
dataset_path = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/datas/ruozhiba"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "cuda", trust_remote_code=True)
model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model_config.cache_max_batch_size = None
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

### 数据预处理
dataset = load_dataset(dataset_path, split="train")  ### load_dataset 直接输入数据集地址即可
def dataset_preprocess(data):
    MAX_LENGTH = 256
    prompt = f"<s>user\n{data["query"]}</s>\n<s>assistant\n"
    response = f"{data["response"]}</s>\n"
    
    prompt_ids = tokenizer(prompt)
    response_ids = tokenizer(response)
    
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
dataset = dataset.shuffle()

### 配置LoRA参数
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules="all-linear",
                         r = 100, lora_alpha=16, lora_dropout=0.6)

### 模型加载Lora配置
model = get_peft_model(model=model, peft_config=lora_config, mixed=False)
model.print_trainable_parameters()
for k, v in model.named_parameters():
    if k.endswith("_tf_layer._layers.47._att_layer._qw.lora_B.default.weight"):
        print(v.shape, v)

### 训练配置与训练
training_argument = TrainingArguments(
    output_dir = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/checkpoint_lora",
    per_device_train_batch_size = 5,
    gradient_accumulation_steps = 2,
    logging_steps = 10,
    save_steps = 500,
    num_train_epochs = 1,
)

trainer = Trainer(
    model = model,
    args = training_argument,
    train_dataset = dataset,
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)
trainer.train()


