from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import pickle
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# 使用 ByteLevel 分词器可以保留空格等边界信息
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=8192*4,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<|endoftext|>"]
)

tokenizer.train( ["./data/train_sampled.txt","./data/TinyStoriesV2-GPT4-train.txt","./data/TinyStoriesV2-GPT4-valid.txt", "./data/alpaca_tokenizer_text.txt"], trainer)

tokenizer.post_processor = processors.TemplateProcessing(
    single="<bos> $A <eos>",
    pair="<bos> $A <eos> $B:1 <eos>:1",
    special_tokens=[
        ("<bos>", tokenizer.token_to_id("<bos>")),
        ("<eos>", tokenizer.token_to_id("<eos>"))
    ],
)

with open("./data/tiny_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)