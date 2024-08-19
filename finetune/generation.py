import logging
import torch
import os
import csv
import logging
import pandas as pd
from dataclasses import dataclass, field
import torch.nn.functional as F
from typing import Optional, Dict, Sequence, Tuple, List
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import transformers
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
from torch.cuda.amp import autocast
sys.path.append('..')
from hybrid_transformer import HierarchicalTransformer
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['WANDB_PROJECT'] = 'Evaluation-dna'
os.environ["WANDB_LOG_MODEL"]="false"
@dataclass
class ModelArguments:
    """Arguments for models."""
    model_name_or_path: Optional[str] = field(default='/path/to/model')
    model_weights_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model name or config path")}, )
    tokenizer_name_or_path: Optional[str] = field(
        default='AIRI-Institute/gena-lm-bert-base',
        metadata={"help": ("The tokenizer name or path")}, )
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=16, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="q,v", metadata={"help": "where to perform LoRA"})

@dataclass
class DataArguments:
    """Arguments for datasets."""
    data_path: List[str] = field(default_factory=lambda: ['/path/to/training/data'], 
                                 metadata={"help": "Path to the training data."})
    
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments for the training loop."""
    num_train_epochs: int = field(default=1, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=12)
    gradient_accumulation_steps: int = field(default=2)
    weight_decay: float = field(default=0.05)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={'help': 'Maximum sequence length. Sequences will be right padded (and possibly truncated).',},)
    flash_attn: Optional[bool] = field(default=False)
    output_dir: str = field(default="output")
    seed: int = field(default=42)
    learning_rate: float = field(default=1e-4)
    lr_scheduler_type: str = field(default="cosine_with_restarts")
    warmup_steps: int = field(default=50)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=1000)
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=1)
    checkpointing: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    find_unused_parameters: bool = field(default=False)
    save_model: bool = field(default=False)
    report_to: Optional[str] = field(default='none')

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict) 


@dataclass
class DataCollatorForDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        
        input_ids = torch.nn.utils.rnn.pad_sequence([s['input_ids'].clone().detach() for s in instances], batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        
        return dict(
            input_ids = input_ids,
            labels = input_ids
        )
    
def preprocess_data(example,tokenizer):
    text = example['text']
    input_ids = tokenizer.encode(text,add_special_tokens=True,truncation=True,max_length=4096)
    example['input_ids'] = input_ids
    return example


def main() -> None:
    """Main training routine."""
    parser = transformers.HfArgumentParser([TrainingArguments, ModelArguments, DataArguments])
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MODEL_NAME_OR_PATH = '/path/to/gena-lm-bert-large-t2t'  

    tokenizer =  AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH,padding_side='left')

    tokenizer.pad_token = tokenizer.sep_token
    

    model = HierarchicalTransformer(
        num_tokens=32000,
        dim = 1024,
        depth = 12,
        dim_head = 64,
        heads = 16,
        seq_len = 150,
        hierarchies=(1,8),
        window_sizes=(16,None)
    )     


    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
   
    pretrained_weights_path='/path/to/pytorch_model.bin'
    pretrained_weights = torch.load(pretrained_weights_path, map_location=device)
    model.load_state_dict(pretrained_weights)
    ##下游任务的数据集

    #获取label
    model.eval()
    model.to(device)
    LL = []
    model.half()
    num_samples = 20
    df = pd.read_csv('/path/to/SKNSH_data/Sknsh_origin_sequence.csv')
    df_subset = df[df['Identifier']=='DHS_natural']
    
    generated_sequences = []

    for sequence in tqdm(df_subset['sequence'],desc='Generating'):
        input_ids =  tokenizer(
                sequence,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )
        input_ids = input_ids['input_ids'].to(device)
        length = input_ids.shape[1]
        truncation_length = int(length*0.3)
        input_ids = input_ids[:,:truncation_length]
        with torch.no_grad():
            for i in range(num_samples):
                # Get embeddings or per-position loss
                with autocast():
                    outputs = model.generate(prompt=input_ids,seq_len=30)
                    
                    generated_tokens = outputs.tolist()
                    generated_texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated_tokens]
                    for text in generated_texts:
                        generated_sequences.append(text)

    df_generated = pd.DataFrame({'GeneratedSequence': generated_sequences})
    


if __name__ == '__main__':
    main()