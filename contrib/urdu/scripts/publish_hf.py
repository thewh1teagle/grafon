"""Private-first Hub publisher with explicit org permission and size guards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

REQUIRED = ["model.safetensors", "config.json", "source_vocab.json", "target_vocab.json", "training_report.json", "README.md"]


def main():
    ap = argparse.ArgumentParser(description="Publish real compact Grafon Urdu weights")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--repo", default="grafon-g2p/ur", help="Use the authorized Grafon organization")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--rights-confirmed", action="store_true")
    ap.add_argument("--allow-existing", action="store_true")
    ap.add_argument("--max-megabytes", type=float, default=60.0)
    args = ap.parse_args()

    directory = Path(args.model_dir).resolve()
    missing = [f for f in REQUIRED if not (directory / f).is_file()]
    if missing:
        raise SystemExit(f"Missing model files: {missing}")
    weights_mb = (directory / "model.safetensors").stat().st_size / 1e6
    if weights_mb > args.max_megabytes:
        raise SystemExit(f"Refusing to publish {weights_mb:.1f} MB weights. Compact size limit is {args.max_megabytes} MB")
    report = json.loads((directory / "training_report.json").read_text(encoding="utf-8"))
    print("Model:", directory)
    print("Weights MB:", round(weights_mb, 2), "parameters:", report.get("parameter_count"))
    print("Intended Hub destination:", args.repo)
    print("Visibility: PRIVATE (will not change existing public visibility)")
    if not args.execute:
        print("Dry run only. Pass --execute after verifying organization write permission.")
        return
    if not args.rights_confirmed:
        raise SystemExit("Publishing requires --rights-confirmed after checking license and attribution")
    api = HfApi()
    who = api.whoami()
    org = args.repo.split("/")[0]
    if "/" not in args.repo:
        raise SystemExit("Repository id must include organization, e.g. grafon-g2p/ur")
    orgs = who.get("orgs", [])
    has_org = any((o.get("name") if isinstance(o,dict) else o) == org for o in orgs)
    if not has_org and who.get("name") != org:
        raise SystemExit(f"Signed in as {who.get('name')!r}; no confirmed membership in {org!r}. "
                         "Ask the organization maintainer for write access or open a pull request.")
    try:
        existing = api.repo_info(repo_id=args.repo, repo_type="model")
    except Exception as e:
        # The create_repo call below is authoritative; transient transport errors still surface there.
        existing = None
    if existing is not None and (not args.allow_existing or not existing.private):
        raise SystemExit("Repository exists or is already public. Refusing to change it without a manual review")
    api.create_repo(repo_id=args.repo, repo_type="model", private=True, exist_ok=args.allow_existing)
    api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(directory),
                      commit_message="Add compact Urdu grapheme-to-IPA model")
    files = set(api.list_repo_files(args.repo, repo_type="model"))
    if not set(REQUIRED).issubset(files):
        raise SystemExit("Upload returned without all required files; inspect Hub repo")
    print("Verified", args.repo, "files:", len(files))


if __name__ == "__main__": main()
