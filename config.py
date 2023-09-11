from pathlib import Path

def get_config():
    return {
        'batch_size': 2048,
        'num_epochs': 20,
        'lr': 10**-4,
        'seq_len': 500,
        'd_model': 512,
        'lang_src': 'en',
        'lang_tgt': 'fr',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': True,
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel'
    }
    
    pass