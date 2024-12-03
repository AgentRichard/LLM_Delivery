from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from peft import AutoPeftModel, PeftModel
from modelscope.utils.constant import Tasks
import warnings
warnings.filterwarnings("ignore")

### pretrain inference
model_dir = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/_cache"
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "cuda",trust_remote_code = True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code = True)

### sft inference
# model_dir = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/sft_checkpoint/checkpoint-2992"
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "cuda",trust_remote_code = True)
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code = True)

### DPO inference
# model_dir = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/checkpoint_dpo"
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "cuda",trust_remote_code = True)
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code = True)

### LoRA inference
#1. 权重不合并
# model_dir = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/_cache"
# lora_dir = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/checkpoint_lora/checkpoint-150"
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "cuda",trust_remote_code = True)
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code = True)
# lora_model = PeftModel.from_pretrained(model=model, model_id=lora_dir, trust_remote_code = True)

# pp = pipeline("text-generation", model=lora_model, tokenizer=tokenizer, trust_remote_code=True)     ## 这个使用的是model.py内部的generation.config
# print(pp("人有几颗心脏"))

#2. 权重合并
# model_dir = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/_cache"
# lora_dir = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/checkpoint_lora/checkpoint-150"
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "cuda",trust_remote_code = True)
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code = True)
# lora_model = PeftModel.from_pretrained(model=model, model_id=lora_dir, trust_remote_code = True)
# lora_model = lora_model.merge_and_unload()
# lora_model.save_pretrained("/root/project/My_projects/MS_PEFT_RLHF_mymodel/checkpoint_lora_merged")

# pp = pipeline("text-generation", model=lora_model, tokenizer=tokenizer, trust_remote_code=True)     ## 这个使用的是model.py内部的generation.config
# print(pp("人有几颗心脏"))

## 通用执行
pp = pipeline(task=Tasks.text_generation, model=model, tokenizer=tokenizer, trust_remote_code=True)     ## 这个使用的是model.py内部的generation.config
print(pp("人有几颗心脏"))