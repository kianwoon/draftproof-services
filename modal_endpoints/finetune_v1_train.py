"""Fine-tune v1 trainer (2026-07-14) — one-shot Modal job, NOT a deployed service.

Trains the desklib-architecture detector (mean-pool + single-logit head, ported
from deberta_large_detector.py) initialized from the PROD checkpoint
(desklib/ai-text-detector-academic-v1.01) on the v1 corpus windows
(3-sentence windows, the exact unit prod scores at inference).

Data in/out via the `draftproof-finetune-v1` Volume:
    modal volume create draftproof-finetune-v1
    modal volume put draftproof-finetune-v1 /tmp/finetune_v1/train_windows.jsonl /train_windows.jsonl
    modal volume put draftproof-finetune-v1 /tmp/finetune_v1/val_windows.jsonl /val_windows.jsonl
    modal run modal_endpoints/finetune_v1_train.py
    modal volume get draftproof-finetune-v1 /model_out /tmp/finetune_v1/model_out
"""
from __future__ import annotations

import modal

BASE_CHECKPOINT = "desklib/ai-text-detector-academic-v1.01"

app = modal.App("draftproof-finetune-v1")
vol = modal.Volume.from_name("draftproof-finetune-v1", create_if_missing=True)


def _bake():
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=BASE_CHECKPOINT, local_dir="/base_model")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers==4.46.3", "torch==2.5.1", "huggingface_hub", "numpy")
    .run_function(_bake)
)


@app.function(gpu="A100", image=image, volumes={"/data": vol}, timeout=3 * 3600)
def train(epochs: float = 1.0, lr: float = 1e-5, batch: int = 16, max_len: int = 256,
          grad_accum: int = 4):
    # batch 64 OOM'd the A100-40GB (deberta-v3-large disentangled attention);
    # 16 x 4 accumulation keeps the same effective batch.
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import json
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

    class DesklibAIDetectionModel(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.init_weights()

        def forward(self, input_ids, attention_mask=None):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last = outputs[0]
            mask = attention_mask.unsqueeze(-1).expand(last.size()).float()
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return {"logits": self.classifier(pooled)}

    class Windows(Dataset):
        def __init__(self, path):
            self.rows = [json.loads(l) for l in open(path) if l.strip()]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]["text"], float(self.rows[i]["label"])

    tok = AutoTokenizer.from_pretrained("/base_model")
    model = DesklibAIDetectionModel.from_pretrained("/base_model").cuda()

    def collate(batch_rows):
        texts, labels = zip(*batch_rows)
        enc = tok(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return enc, torch.tensor(labels)

    train_dl = DataLoader(Windows("/data/train_windows.jsonl"), batch_size=batch,
                          shuffle=True, collate_fn=collate, num_workers=4)
    val_dl = DataLoader(Windows("/data/val_windows.jsonl"), batch_size=batch,
                        collate_fn=collate, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = int(len(train_dl) * epochs)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, total_steps)
    scaler = torch.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()

    def evaluate():
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for enc, y in val_dl:
                enc = {k: v.cuda() for k, v in enc.items()}
                with torch.amp.autocast("cuda"):
                    logits = model(enc["input_ids"], attention_mask=enc["attention_mask"])["logits"].squeeze(-1)
                pred = (torch.sigmoid(logits) > 0.5).float().cpu()
                correct += (pred == y).sum().item()
                n += len(y)
        model.train()
        return correct / max(n, 1)

    model.train()
    step = 0
    micro = 0
    opt.zero_grad()
    for _epoch in range(int(epochs) if epochs >= 1 else 1):
        for enc, y in train_dl:
            if step >= total_steps:
                break
            enc = {k: v.cuda() for k, v in enc.items()}
            y = y.cuda()
            with torch.amp.autocast("cuda"):
                logits = model(enc["input_ids"], attention_mask=enc["attention_mask"])["logits"].squeeze(-1)
                loss = loss_fn(logits, y) / grad_accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            step += 1
            if step % 400 == 0:
                print(f"step {step}/{total_steps} loss {loss.item() * grad_accum:.4f}", flush=True)
            if step % 1600 == 0:
                print(f"  val acc: {evaluate():.4f}", flush=True)
    acc = evaluate()
    print(f"FINAL val acc: {acc:.4f}", flush=True)
    model.save_pretrained("/data/model_out")
    tok.save_pretrained("/data/model_out")
    vol.commit()
    return {"val_acc": acc, "steps": step}


@app.local_entrypoint()
def main():
    print(train.remote())
