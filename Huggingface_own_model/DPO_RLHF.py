from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from trl import DPOConfig, DPOTrainer
from modelscope.msdatasets import MsDataset
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

model_path = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/_cache"
config = AutoConfig.from_pretrained(model_path, trust_remote_code = True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)
config.cache_max_batch_size = None
model = AutoModelForCausalLM.from_pretrained(model_path, config = config, trust_remote_code=True)

dataset = load_dataset("/root/project/My_projects/MS_PEFT_RLHF_mymodel/datas/dpo_chinese_pair", split="train")

def preprocess_dataset(data):
    return {
        "prompt" : f"<s>user\n{data["prompt"]}</s><s>assistant\n",      ### 提示词一定到ai要回答的地方
        "chosen" : f"{data["chosen"]}</s>",
        "rejected" : f"{data["rejected"]}</s>"
    }
    ### 如果不考虑参考模型，一条数据进去会生成2条数据，反之生成4条数据
    
dataset = dataset.map(preprocess_dataset, remove_columns=dataset.column_names)

dataset = dataset.shuffle()

training_args = DPOConfig(
    output_dir = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/checkpoint_dpo",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_train_epochs = 1,
    save_steps = 1000
    # optim="paged_adamw_32bit"
)

trainer = DPOTrainer(
    model = model,
    tokenizer = tokenizer,
    args = training_args,
    train_dataset = dataset,
    max_prompt_length = 512,
    max_length = 512,
    max_target_length = 512
)

trainer.train()

