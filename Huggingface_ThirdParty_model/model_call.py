from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from peft import PeftModel
from modelscope.utils.constant import Tasks
import warnings
warnings.filterwarnings("ignore")

## pretrain inference
# model_dir = "/root/project/My_projects/MS_PEFT_RLHF_qwen_0.5B_instruct/_cache/Qwen/Qwen2___5-0___5B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "cuda")
# tokenizer = AutoTokenizer.from_pretrained(model_dir)

## Lora inference 
#1. 非合并
# model_dir = "/root/project/My_projects/MS_PEFT_RLHF_qwen_0.5B_instruct/_cache/Qwen/Qwen2___5-0___5B-Instruct"
# lora_dir = "/root/project/My_projects/MS_PEFT_RLHF_qwen_0.5B_instruct/checkpoint_lora/checkpoint-150"
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "cuda")
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# lora_model = PeftModel.from_pretrained(model=model, model_id=lora_dir)
# pp = pipeline("text-generation", model=lora_model, tokenizer=tokenizer)     ## 这个使用的是model.py内部的generation.config
# print(pp("你是谁?"))

#2.合并
# model_dir = "/root/project/My_projects/MS_PEFT_RLHF_qwen_0.5B_instruct/_cache/Qwen/Qwen2___5-0___5B-Instruct"
# lora_dir = "/root/project/My_projects/MS_PEFT_RLHF_qwen_0.5B_instruct/checkpoint_lora/checkpoint-150"
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "cuda")
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# lora_model = PeftModel.from_pretrained(model=model, model_id=lora_dir)
# lora_model = lora_model.merge_and_unload()
# lora_model.save_pretrained("/root/project/My_projects/MS_PEFT_RLHF_qwen_0.5B_instruct/checkpoint_lora_merged")

## Qlora inference same as Lora
#1. 非合并
# model_dir = "/root/project/My_projects/MS_PEFT_RLHF_qwen_0.5B_instruct/_cache/Qwen/Qwen2___5-0___5B-Instruct"
# lora_dir = "/root/project/My_projects/MS_PEFT_RLHF_qwen_0.5B_instruct/checkpoint_qlora/checkpoint-500"
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "cuda")
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# lora_model = PeftModel.from_pretrained(model=model, model_id=lora_dir)
# pp = pipeline("text-generation", model=lora_model, tokenizer=tokenizer)     ## 这个使用的是model.py内部的generation.config
# print(pp("你是谁?"))

#2.合并
model_dir = "/root/project/My_projects/MS_PEFT_RLHF_qwen_0.5B_instruct/checkpoint_qlora_merged"
tokenizer_dir = "/root/project/My_projects/MS_PEFT_RLHF_qwen_0.5B_instruct/_cache/Qwen/Qwen2___5-0___5B-Instruct"
lora_dir = "/root/project/My_projects/MS_PEFT_RLHF_qwen_0.5B_instruct/checkpoint_qlora/checkpoint-500"
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "cuda")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
lora_model = PeftModel.from_pretrained(model=model, model_id=lora_dir)
lora_model = lora_model.merge_and_unload()
# lora_model.save_pretrained("/root/project/My_projects/MS_PEFT_RLHF_qwen_0.5B_instruct/checkpoint_qlora_merged")
pp = pipeline(task=Tasks.text_generation, model=lora_model, tokenizer=tokenizer)     ## 这个使用的是model.py内部的generation.config
print(pp("你是谁?"))

# # general call
# pp = pipeline(task=Tasks.text_generation, model=model, tokenizer=tokenizer)     ## 这个使用的是model.py内部的generation.config
# print(pp("你是谁?"))