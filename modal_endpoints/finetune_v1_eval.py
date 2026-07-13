"""Score the frozen eval pack with the fine-tuned v1 checkpoint on GPU.
Reads /data/model_out + /data/eval_pack.jsonl, writes /data/eval_scores.jsonl.
Same 3-sentence windowing + mean aggregation as the bake-off harness."""
import modal

app = modal.App("draftproof-finetune-v1-eval")
vol = modal.Volume.from_name("draftproof-finetune-v1")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("transformers==4.46.3", "torch==2.5.1", "numpy"))

@app.function(gpu="L4", image=image, volumes={"/data": vol}, timeout=3600)
def score():
    import json, re, torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

    class DesklibAIDetectionModel(PreTrainedModel):
        config_class = AutoConfig
        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None):
            out = self.model(input_ids, attention_mask=attention_mask)[0]
            m = attention_mask.unsqueeze(-1).expand(out.size()).float()
            pooled = (out*m).sum(1)/m.sum(1).clamp(min=1e-9)
            return {"logits": self.classifier(pooled)}

    tok = AutoTokenizer.from_pretrained("/data/model_out")
    mdl = DesklibAIDetectionModel.from_pretrained("/data/model_out").cuda().eval()

    SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
    def split_sentences(t):
        t = re.sub(r"\s+", " ", t).strip()
        return [s.strip() for s in SENT.split(t) if s.strip()]

    def win(sents, size=3, step=1):
        if len(sents) <= size: return [" ".join(sents)] if sents else []
        return [" ".join(sents[i:i+size]) for i in range(0, len(sents)-size+1, step)]

    def score_windows(ws, batch=64):
        out=[]
        for i in range(0, len(ws), batch):
            enc = tok(ws[i:i+batch], padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k:v.cuda() for k,v in enc.items()}
            with torch.no_grad(), torch.amp.autocast("cuda"):
                lg = mdl(enc["input_ids"], attention_mask=enc["attention_mask"])["logits"].squeeze(-1)
            out.extend(torch.sigmoid(lg).float().cpu().tolist())
        return out

    n=0
    with open("/data/eval_scores.jsonl","w") as outf:
        for line in open("/data/eval_pack.jsonl"):
            line=line.strip()
            if not line: continue
            d=json.loads(line)
            sents=split_sentences(d.get("text") or "")
            if not sents: continue
            ws=win(sents)
            probs=score_windows(ws)
            # per-sentence mean of covering windows (size3 step1)
            per=[[] for _ in sents]
            for wi,p in enumerate(probs):
                for si in range(wi, min(wi+3, len(sents))):
                    per[si].append(p)
            scores=[sum(x)/len(x) for x in per if x]
            outf.write(json.dumps({"id":d["id"],"pop":d["pop"],"family":d.get("family"),
                "attack":d.get("attack"),"scores":[round(s,6) for s in scores]})+"\n")
            n+=1
            if n%100==0: print(n, flush=True)
    vol.commit()
    print("scored", n, flush=True)

@app.local_entrypoint()
def main():
    score.remote()
