
import logging 
import torch
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any,Sequence
import transformers
from tqdm import tqdm
import glob
from datasets import concatenate_datasets,load_from_disk
from transformers import Trainer
sys.path.append('..')
from hybrid_transformer import HierarchicalTransformer
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for models."""
    model_config_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model name or config path")}, )
    tokenizer_name_or_path: Optional[str] = field(
        default='AIRI-Institute/gena-lm-bert-base',
        metadata={"help": ("The tokenizer name or path")}, )

@dataclass
class DataArguments:
    """Arguments for datasets."""
    data_path: List[str] = field(default_factory=lambda: ['/path/to/training/data'], 
                                 metadata={"help": "Path to the training data."})
    valid_data_path: str = field(default='')
    data_impl: str = field(default='mmap',metadata={"help":('type of dataset produced by preprocess_data.py')})
    data_skip_warmup: bool = field(default=False,
                                   metadata={"help": "Skip dataset warmup."})
    
    data_name: str = field(default='', metadata={"help": "used to save/load samples mapping .npy index"})
    
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments for the training loop."""
    num_train_epochs: int = field(default=2, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=12)
    gradient_accumulation_steps: int = field(default=2)
    weight_decay: float = field(default=0.01)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={'help': 'Maximum sequence length. Sequences will be right padded (and possibly truncated).',},)
    flash_attn : Optional[bool] = field(default=False)
    output_dir: str = field(default="debug")
    seed: int = field(default=42)
    learning_rate: float = field(default=2e-5)
    lr_scheduler_type: str = field(default="cosine_with_restarts")
    warmup_steps: int = field(default=2000)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=1000)
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=1)
    checkpointing: bool = field(default=False)
    report_to: Optional[str] = field(default='none')
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict) 
        
class RNADataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: any,
                 mode: str,  # 'mode' seems unused; you might want to utilize it if needed.
                 kmer: int = -1):  # 'kmer' seems unused; you might want to utilize it if needed.

        super(RNADataset, self).__init__()
        self.tokenizer = tokenizer
        with open(data_path,'r') as file:
            self.texts = file.readlines()
        
        # Use map to apply tokenizer to each text in self.texts
        tokenized_texts = [self._tokenize_text(text) for text in tqdm(self.texts, desc="Tokenizing")]

        self.input_ids = [item['input_ids'].squeeze(0) for item in tokenized_texts]
        

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Helper function to tokenize a single text."""
        output = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True
        )
        return output

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.input_ids[i])
@dataclass
class DataCollatorForDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(s['input_ids']) for s in instances], batch_first=True,
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
def load_all_batches(base_path):
    # 使用glob模块找到所有批次的文件夹
    batch_folders = glob.glob(os.path.join(base_path, 'processed_dataset_batch_*'))
    all_datasets = []

    for batch_folder in batch_folders:
        if os.path.exists(batch_folder):
            batch_dataset = load_from_disk(batch_folder)
            all_datasets.append(batch_dataset)
        else:
            print(f"Batch folder {batch_folder} not found.")

    if all_datasets:
        # 合并所有批次的数据集
        merged_dataset = concatenate_datasets(all_datasets)
        return merged_dataset
    else:
        return None
    
def main() -> None:
    """Main training routine."""
    parser = transformers.HfArgumentParser([TrainingArguments, ModelArguments, DataArguments])
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    MODEL_NAME_OR_PATH = '../model_and_data/gena-lm-bert-large-t2t'  
    tokenizer =  AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH,model_max_length=training_args.model_max_length,use_fast=True)
    tokenizer.pad_token = tokenizer.sep_token
    ###这里先被注释掉
    model = HierarchicalTransformer(
        num_tokens=32000,
        dim = 1024,
        depth = 12,
        dim_head = 64,
        heads = 12,
        seq_len = training_args.model_max_length,
        hierarchies=(1,8),
        window_sizes=(64,None)
        )   
    
 
    processed_dataset_path = '/processed_dataset_20000'
    if os.path.exists(processed_dataset_path):
        merged_dataset = load_from_disk(processed_dataset_path)

    train_dataset = merged_dataset
    print('Loading saved datasets...')

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    print("Pretrained Model Layers:")
    
    data_collator = DataCollatorForDataset(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == '__main__':
    main()