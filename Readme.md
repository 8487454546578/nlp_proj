# Readme

221850088殷嘉欣 

221830083韩子皓

【注：advanced目录下仅仅是将进阶任务的代码单独放了出来，在llama文件夹中也有并且要在llama中才能运行相关部分】

微调的人工评分数据结果在model_eval_samples.csv中



## 数据

应放置在llama同级目录的data文件夹中

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
PYTHONPATH=. python ./llama/scripts/finetune_eval.py

PYTHONPATH=. python ./llama/scripts/train_reward_base_for_eval.py
PYTHONPATH=. python ./llama/scripts/train_reward_base_for_ppo.py

PYTHONPATH=. python ./llama/scripts/train_ppo_lora.py
PYTHONPATH=. python ./llama/scripts/train_ppo_constrancy.py
PYTHONPATH=. python ./llama/scripts/train_ppo_consistency.py

#预测rlhf输出：
PYTHONPATH=. python ./llama/scripts/rlhf_eval.py

```



## 日志

将终端的输出复制到了result.txt中

