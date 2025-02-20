import os
import json
import importlib
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List
import pandas as pd
import torch.nn as nn
import torch
import transformers
import sklearn
import numpy as np
import sys
sys.path.append('..')
from hybrid_transformer import HierarchicalTransformer
import scipy
from torch.utils.data import Dataset
from transformers import AutoTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['WANDB_PROJECT'] = 'Evaluation-dna'
os.environ["WANDB_LOG_MODEL"]="false"
@dataclass
class ModelArguments:
    """Arguments for models."""
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
    lora_target_modules: str = field(default="to_qkv", metadata={"help": "where to perform LoRA"})
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


class Variant_FT_Dataset(Dataset):
    def __init__(self, data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer,
                 split: str = "train"):

        self.split = split
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.sequences = self.data['ref_sequence'].tolist()
        self.labels = self.data['label'].tolist()
    
        output = self.tokenizer(
            self.sequences,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]                                
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(text=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("text", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            text=input_ids,
            labels=labels,     
        )

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    probabilities = scipy.special.softmax(logits, axis=-1)
    positive_probs = probabilities[:, 1][valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="binary", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="binary", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="binary", zero_division=0
        ),
        "auroc": sklearn.metrics.roc_auc_score(
            valid_labels, positive_probs
        )
    }
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma
    
class HierarchicalTransformerWitnClassifier(nn.Module):
    def __init__(self,base_model,num_classes):
        super(HierarchicalTransformerWitnClassifier, self).__init__(
        )
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False  
        self.fc1 = nn.Linear(base_model.dim,512)  
        self.fc2 = nn.Linear(512, 2)
        self.classifier = nn.Linear(base_model.dim,2)
        self.dropout = nn.Dropout(0.2)
        self.num_labels = num_classes
        self.batch_norm = nn.BatchNorm1d(base_model.dim)
        self.norm = RMSNorm(base_model.dim)
        self.layer_norm = nn.LayerNorm(base_model.dim)
    def forward(self,text,labels):
        
        embeddings = self.base_model(text,labels,return_loss = False,return_hierarchical_embeds = True)
        
        embeddings = embeddings[0]  + embeddings[1]

        averaged_mean_embeddings = torch.mean(embeddings,dim=1)
        averaged_embeddings = averaged_mean_embeddings
        normalized_embeddings = self.batch_norm(averaged_embeddings)
        pooled_output =  self.dropout(normalized_embeddings)
        fc1_output = F.relu(self.fc1(pooled_output))
        logits = self.fc2(fc1_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
            
            return {'logits': logits, 'loss': loss}
        else:
            return {'logits': logits}
"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
   
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)

def get_cls_by_name(name: str) -> type:
    module_name, cls_name = name.split(':')
    return getattr(importlib.import_module(module_name), cls_name)

def train():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer =  AutoTokenizer.from_pretrained('../model_and_data/gena-lm-bert-large-t2t',
                                               padding_side="left",
                                               model_max_length=training_args.model_max_length)
    
    base_model = HierarchicalTransformer(
        num_tokens=32000,
        dim = 1024,
        depth = 12,
        dim_head = 64,
        heads = 16,
        seq_len = training_args.model_max_length,
        hierarchies=(1,8),
        window_sizes=(128,None)
        )     
    n_params = sum({p.data_ptr(): p.numel() for p in base_model.parameters()}.values())
    
    print(f" base model - Total size={n_params/2**20:.2f}M params")
    num_classes = 2
    model = HierarchicalTransformerWitnClassifier(base_model,num_classes).to(device)

    pretrained_weights_path = model_args.model_weights_path
    pretrained_weights = torch.load(pretrained_weights_path, map_location=device)
    

    pre_weights = {name: param.clone() for name, param in model.named_parameters()}
    model.base_model.load_state_dict(pretrained_weights)

    changed_params = [name for name, param in model.named_parameters() if pre_weights[name].ne(param).any()]
    if changed_params:
        print("Loaded pretrained weights have caused changes in the model for parameters:")
    
    tokenizer.pad_token = tokenizer.sep_token
    
    # define datasets and data collator
    train_data_path = '../model_and_data/train_eqtl_sequences_10k.csv'
    test_data_path = '../model_and_data/test_eqtl_sequences_10k.csv'
    train_dataset = Variant_FT_Dataset(tokenizer=tokenizer, 
                                      data_path=train_data_path, 
                                      split='train')
                                      
    
    test_dataset = Variant_FT_Dataset(tokenizer=tokenizer, 
                                     data_path=test_data_path, 
                                     split='test')
    
    print(len(train_dataset[0]['text']))
    print(train_dataset[0]['text'])
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    
    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=test_dataset,
                                   data_collator=data_collator)
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)




if __name__ == "__main__":
    train()
