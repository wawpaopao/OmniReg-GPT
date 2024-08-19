import os
import json
import importlib
from dataclasses import dataclass, field
import torch.nn.functional as F
from typing import Optional, Dict, Sequence, Tuple, List
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import transformers
import sklearn
import numpy as np
import h5py
from scipy.stats import pearsonr
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
from scipy import sparse
sys.path.append('..')

from hybrid_transformer import HierarchicalTransformer

from torch.utils.data import Dataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['WANDB_PROJECT'] = 'Evaluation-dna'
os.environ["WANDB_LOG_MODEL"] = "false"

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

class ATACDataset(Dataset):
    def __init__(self, seqs_file, labels_file, tokenizer):
        # Load the sequence data
        with h5py.File(seqs_file, 'r') as f:
            seqs = f['X'][()]  # Assuming the sequences are stored under the key 'X'
        self.encoding_to_nt = {0: "A", 1: "C", 2: "G", 3: "T"}
        self.seqs = ["".join([self.encoding_to_nt[nt] for nt in seq]) for seq in seqs]
        # Load the labels
        self.labels = sparse.load_npz(labels_file).toarray()
        output = tokenizer(
            self.seqs,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=250,
        )
        self.input_ids = output["input_ids"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.input_ids[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = torch.where(label > 0, torch.tensor(1.0), torch.tensor(0.0))
        return dict(input_ids=seq, 
                    labels=label)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.stack(labels)
        return dict(
            input_ids=input_ids,
            labels=labels,
        )

from sklearn.metrics import roc_auc_score

def auROC(y_true, y_pred):
    temp = []
    for i in range(y_true.shape[1]):
        try:
            ROC = roc_auc_score(y_true[:, i], y_pred[:, i], average='micro', sample_weight=None)
        except ValueError:
            ROC = 0
        temp.append(ROC)
    return np.mean(temp)

def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = (logits > 0).astype(int)
    logits = torch.tensor(logits)
    prob = F.sigmoid(logits)
    prob = prob.detach().cpu().numpy()
    
    micro_roc_auc = roc_auc_score(labels, prob, average='micro')
    macro_roc_auc = roc_auc_score(labels, prob, average='macro')
    torch.save(prob, 'scATAC_prob.pt')
    torch.save(labels, 'scATAC_labels.pt')
    return {
        "f1": sklearn.metrics.f1_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "precision": sklearn.metrics.precision_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            labels, predictions, average="micro", zero_division=0
        ),
        'micro_roc_auc': micro_roc_auc,
        'macro_roc_auc': macro_roc_auc,
    }

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma

class TransposeLayer(nn.Module):
    """A layer that transposes the input."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.transpose(x, 1, 2)

class HierarchicalTransformerWitnClassifier(nn.Module):
    def __init__(self, base_model):
        super(HierarchicalTransformerWitnClassifier, self).__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(1024, 2034)#base_model.dim to num of label
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.BatchNorm1d(1024)
        
    def forward(self, input_ids, labels):
        embeddings = self.base_model(input_ids, labels, return_loss=False, return_hierarchical_embeds=True)
        embeddings = embeddings[0] + embeddings[1]
        
        averaged_embeddings = torch.mean(embeddings, dim=1)
        normalized_embeddings = self.norm(averaged_embeddings)
        pooled_output = self.dropout(normalized_embeddings)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return {'logits': logits, 'loss': loss}
        else:
            return {'logits': logits}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)

def get_cls_by_name(name: str) -> type:
    module_name, cls_name = name.split(':')
    return getattr(importlib.import_module(module_name), cls_name)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    MODEL_NAME_OR_PATH = '/path/to/gena-lm-bert-large-t2t' # load tokenizer of gena-lm-bert-large-t2t

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, padding_side='left')
    tokenizer.pad_token = tokenizer.sep_token
   
    # define datasets and data collator
    train_dataset = torch.load('/path/to/train_dataset.pt')
    test_dataset = torch.load('/path/to/test_dataset.pt')
    

    # train_dataset = ATACDataset('/path/to/sc_ATAC/basenji/output/train_seqs.h5',
    #                         '/path/to/sc_ATAC/basenji/output/m_train.npz',
    #                         tokenizer)
    # test_dataset = ATACDataset('/path/to/sc_ATAC/basenji/output/test_seqs.h5',
    #                         '/path/to/sc_ATAC/basenji/output/m_test.npz',
    #                         tokenizer)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    base_model = HierarchicalTransformer(
        num_tokens=32000,
        dim=1024,
        depth=12,
        dim_head=64,
        heads=16,
        seq_len=training_args.model_max_length,
        hierarchies=(1, 8),
        window_sizes=(64, None)
    )     

    n_params = sum({p.data_ptr(): p.numel() for p in base_model.parameters()}.values())
    
    print(f"base model - Total size={n_params/2**20:.2f}M params")

    model = HierarchicalTransformerWitnClassifier(base_model).to(device)
  
    pretrained_weights_path = model_args.model_weights_path
    pretrained_weights = torch.load(pretrained_weights_path, map_location=device)
    model.base_model.load_state_dict(pretrained_weights, strict=False)
    #load pretrained model weights
    
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    train()