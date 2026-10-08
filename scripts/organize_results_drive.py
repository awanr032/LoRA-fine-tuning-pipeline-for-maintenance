# Run in Colab: copies (never moves) every result file into versioned folders and writes a manifest.
import os, json, shutil, hashlib, glob, datetime
if not os.path.exists("/content/drive/MyDrive"):
    from google.colab import drive; drive.mount("/content/drive")

BASE = "/content/drive/MyDrive/mistral_ft"
DEST = f"{BASE}/paper_results"
PLAN = {
    "v1_teacher_pilots": ["distill/pilot_indices.json", "distill/few_shot_indices.json",
                          "distill/prompt_pool_indices.json", "distill/pilot_results_selfhosted.json"],
    "v2_teacher_labels": ["distill/silver_selfhosted_nothink.jsonl", "distill/silver_selfhosted.jsonl",
                          "distill/silver_deepseek.jsonl"],
    "v3_data_and_splits": ["data/splits.json", "data/official_split.json", "data/train.jsonl", "data/val.jsonl",
                           "data/test.jsonl", "data_gold/*.jsonl", "data_distilled_860/*"],
    "v3_internal_eval": ["results/predictions_distilled_qwen_nothink_v1.jsonl",
                         "results/metrics_distilled_qwen_nothink_v1.json"],
    "v4_official_eval": ["results/*_official.jsonl", "results/metrics_*_official.json"],
    "v5_controlled_comparisons": ["results/comparison_*.json"],
}
ADAPTERS = ["distilled_qwen_nothink_v1", "gold_official_v1", "distilled_860_v1"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(), "versions": {}, "adapters": {}, "missing": []}
for version, patterns in PLAN.items():
    os.makedirs(f"{DEST}/{version}", exist_ok=True)
    files = []
    for pat in patterns:
        hits = [p for p in glob.glob(f"{BASE}/{pat}") if os.path.isfile(p)]
        if not hits:
            manifest["missing"].append(pat)
        for src in hits:
            rel = os.path.relpath(src, BASE).replace("/", "__")
            dst = f"{DEST}/{version}/{rel}"
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            files.append({"file": rel, "source": os.path.relpath(src, BASE),
                          "bytes": os.path.getsize(dst), "sha256": sha256(dst)})
    manifest["versions"][version] = files
    print(f"{version}: {len(files)} files")

for run in ADAPTERS:   # adapters are large: record location and fingerprint, don't copy
    cfg = f"{BASE}/outputs/{run}/final_adapter/adapter_config.json"
    manifest["adapters"][run] = {"path": f"outputs/{run}/final_adapter",
                                 "adapter_config_sha256": sha256(cfg) if os.path.exists(cfg) else None}

for extra in ["RESULTS_REGISTRY.md", "all_results.csv"]:   # upload these two files to Colab first (Files panel)
    if os.path.exists(f"/content/{extra}"):
        shutil.copy2(f"/content/{extra}", f"{DEST}/{extra}")

json.dump(manifest, open(f"{DEST}/manifest.json", "w"), indent=2)
print("missing patterns (fine if a file was never created):", manifest["missing"])
print("done:", DEST)
