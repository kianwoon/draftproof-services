"""Download the DeBERTa checkpoint tarball from R2 and extract it onto the HF volume.

The worker's runtime cannot reach huggingface.co (egress blocked), so the checkpoint tarball is
staged in R2 (reachable — same store reports use) and pulled onto the persistent /app/hf_cache
volume at first boot. After that, scans load it local_files_only. Mirrors the gpt2/MiniLM
"save to volume" pattern; the tarball is the save_pretrained() output of
fakespot-ai/roberta-base-ai-text-detection-v1.

Security (addresses the 3 findings on the first version):
- INTEGRITY: verifies an expected SHA256 + byte size BEFORE extracting, so a tampered/swapped
  tarball is rejected. If the model is re-uploaded, update EXPECTED_SHA256 + EXPECTED_SIZE.
- RESOURCE EXHAUSTION: streams to a temp file on disk and hashes in 8MB chunks — never holds
  the ~464MB tarball in RAM.
- PATH TRAVERSAL: extraction is allowlisted to known model filenames only, each flattened to a
  basename (rejects any member with a path separator, absolute path, symlink, or unknown name);
  --out is constrained to live under HF_HOME.

Usage: python download_deberta.py --out /app/hf_cache/deberta-fakespot
"""
import argparse
import hashlib
import os
import sys
import tarfile
import tempfile

R2_KEY = "models/deberta-fakespot.tar.gz"
EXPECTED_SHA256 = "2c275f553b7f4e319f00565d11090a6e9280dbc8221e8a2b94fc6dfe34520e65"
EXPECTED_SIZE = 464207993

# Only these model files may be extracted (allowlist = strongest path-traversal guard).
ALLOWED_BASENAMES = {
    "config.json", "merges.txt", "model.safetensors", "pytorch_model.bin",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="target dir on the volume")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    hf_home = os.path.abspath(os.environ.get("HF_HOME", "/app/hf_cache"))
    if not (out == hf_home or out.startswith(hf_home + os.sep)):
        print(f"[deberta-download] refusing --out outside HF volume: {out} (HF_HOME={hf_home})",
              file=sys.stderr)
        return 4

    for k in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"):
        if not os.environ.get(k):
            print(f"[deberta-download] missing env {k} — cannot fetch from R2", file=sys.stderr)
            return 2

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["R2_BUCKET_NAME"]

    os.makedirs(out, exist_ok=True)

    # Stream to a temp file on disk (avoids holding the full tarball in RAM).
    print(f"[deberta-download] fetching s3://{bucket}/{R2_KEY} ...", flush=True)
    tmp_path = tempfile.mktemp(prefix="deberta-", suffix=".tar.gz")
    try:
        with open(tmp_path, "wb") as tf:
            s3.download_fileobj(bucket, R2_KEY, tf)

        # Hash in 8MB chunks (constant RAM) + size check.
        h = hashlib.sha256()
        total = 0
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
                total += len(chunk)
        digest = h.hexdigest()
        if total != EXPECTED_SIZE or digest != EXPECTED_SHA256:
            print(f"[deberta-download] INTEGRITY FAILED: size={total} (expected {EXPECTED_SIZE}), "
                  f"sha256={digest}", file=sys.stderr)
            return 3
        print(f"[deberta-download] integrity OK (size={total}, sha256={digest[:12]}…)", flush=True)

        # Allowlisted extraction: regular files only, basename must be a known model file.
        print(f"[deberta-download] extracting to {out} ...", flush=True)
        with tarfile.open(tmp_path, mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                if member.name != os.path.basename(member.name) or member.name not in ALLOWED_BASENAMES:
                    print(f"[deberta-download] refusing tar entry: {member.name!r}", file=sys.stderr)
                    continue
                member.name = member.name  # already a clean basename
                tar.extract(member, path=out)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    print(f"[deberta-download] done: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
