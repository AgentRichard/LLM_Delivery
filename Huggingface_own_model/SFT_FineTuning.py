from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

model_path = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/_cache"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code = True)
config.cache_max_batch_size = None
model = AutoModelForCausalLM.from_pretrained(model_path, config = config, trust_remote_code=True)

dataset = load_dataset(path="/root/project/My_projects/MS_PEFT_RLHF_mymodel/datas/ruozhiba", split="train")
print(dataset)
def preprocess_datasets(data):
    return {"text": f"<s>user\n{data["query"]}</s><s>assistant\n{data["response"]}</s>"}
dataset = dataset.map(preprocess_datasets, remove_columns=dataset.column_names)
dataset.shuffle()

response_template = "<s>assistant\n"    ## 这个步骤是 仅有response_template后面的内容参与损失计算
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

training_args = SFTConfig(
    output_dir="/root/project/My_projects/MS_PEFT_RLHF_mymodel/checkpoint_sft",
    dataset_text_field="text",
    max_seq_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    save_steps=1000,
    logging_steps=10,
    # optim="paged_adamw_32bit"
)

trainer = SFTTrainer(
    model = model,
    tokenizer=tokenizer,
    args = training_args,
    train_dataset=dataset,
    data_collator=collator,
)

trainer.train()
