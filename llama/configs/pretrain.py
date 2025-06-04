PRETRAIN_CONFIG = {
    'train_data_path': "./data/TinyStoriesV2-GPT4-train.txt",
    'valid_data_path': "./data/TinyStoriesV2-GPT4-valid.txt",
    'tokenizer_path': "./data/tiny_tokenizer.pkl",
    'block_size': 1024,
    'batch_size': 4,
    'lr': 3e-4,
    'epochs': 30,
    'save_dir': "checkpoints",
    'pic_dir': "pic",
    'best_model_name': "tiny_transformer_best.pth",
    'early_stop_ppl': 30,
    'model_dim': 1024,
    'n_heads': 32,
    'n_layers': 16
}
