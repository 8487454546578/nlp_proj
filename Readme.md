# Readme

## 数据

放在llama同级目录的data文件夹中

![image-20250626200223917](C:\Users\a little star\AppData\Roaming\Typora\typora-user-images\image-20250626200223917.png)

其中用于训练奖励模型的数据在放置好 `PKU-SafeRLHF`的`train.jsonl`和`test.jsonl`后运行`llama/scripts/make_rlhf_dataset.py`

即可生成，其余数据从网络上可以下载

## 环境

安装以下包：

```
pytorch
peft
tokenizers
transformers
```

## 执行顺序

```
python /data/bead/NLP/pro/nlp_proj/llama/scripts/train_tokenizer.py
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12345 /data/bead/NLP/pro/nlp_proj/llama/scripts/pretrain_main.py

PYTHONPATH=. torchrun --nproc_per_node=8 /data/bead/NLP/pro/nlp_proj/llama/scripts/finetune_main.py
PYTHONPATH=. python /data/bead/NLP/pro/nlp_proj/llama/scripts/finetune_eval.py
PYTHONPATH=. python /data/bead/NLP/pro/nlp_proj/llama/scripts/train_reward_base_for_eval.py
PYTHONPATH=. python /data/bead/NLP/pro/nlp_proj/llama/scripts/train_reward_base_for_ppo.py
PYTHONPATH=. python /data/bead/NLP/pro/nlp_proj/llama/scripts/train_ppo_lora.py
PYTHONPATH=. python /data/bead/NLP/pro/nlp_proj/llama/scripts/train_ppo_constrancy.py
PYTHONPATH=. python /data/bead/NLP/pro/nlp_proj/llama/scripts/train_ppo_consistency.py

```

