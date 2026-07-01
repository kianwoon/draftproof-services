"""Download the DeBERTa checkpoint tarball from R2 and extract it onto the HF volume.

WHY: the worker's runtime cannot reach huggingface.co (egress blocked), so the checkpoint can't
be lazy-loaded or warm-downloaded from HF like gpt2/MiniLM. Instead the tarball is staged in R2
(same object store the worker already uses for reports — reachable) and pulled onto the
persistent /app/hf_cache volume at first boot. After that, scans load it local_files_only.

The tarball is the save_pretrained() output of fakespot-ai/roberta-base-ai-text-detection-v1,
uploaded to s3://<R2_BUCKET>/models/deberta-fakespot.tar.gz (see poc tooling / local upload).

Usage: python download_deberta.py --out /app/hf_cache/deberta-fakespot
"""
import argparse
import io
import os
import sys
import tarfile

R2_KEY = "models/deberta-fakespot.tar.gz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="target dir on the volume")
    args = ap.parse_args()

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

    os.makedirs(args.out, exist_ok=True)
    buf = io.BytesIO()
    bucket = os.environ["R2_BUCKET_NAME"]
    print(f"[deberta-download] fetching s3://{bucket}/{R2_KEY} ...", flush=True)
    s3.download_fileobj(bucket, R2_KEY, buf)
    n = buf.getbuffer().nbytes
    if n < 1_000_000:
        print(f"[deberta-download] ERROR: tarball too small ({n} bytes) — aborting", file=sys.stderr)
        return 3
    buf.seek(0)
    print(f"[deberta-download] extracting to {args.out} ({n / 1e6:.1f} MB) ...", flush=True)
    # Safe extraction: only regular files, flattened to basename (no path traversal).
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            member.name = os.path.basename(member.name)
            tar.extract(member, path=args.out)
    print(f"[deberta-download] done: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
