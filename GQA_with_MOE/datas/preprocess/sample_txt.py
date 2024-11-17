import json

with open("/root/project/My_projects/Transformer/GQA_with_MOE/datas/preprocess/sample.jsonl", mode="r", encoding="utf-8") as fr:
    with open("/root/project/My_projects/Transformer/GQA_with_MOE/datas/preprocess/sample.txt", mode="a+",encoding="utf-8") as fw:
        for sentence in fr.readlines():
            text = json.loads(sentence)
            fw.write(text["text"])
            
        