"""Optional private-first dataset upload, with explicit licensing and org checks."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi
from grafon_urdu.text import read_tsv, sha256

CARD = """---
language:
- ur
task_categories:
- text-to-speech
- text-generation
tags:
- g2p
- urdu
- ipa
- grapheme-to-phoneme
pretty_name: Grafon Urdu sentence and IPA corpus
---

# Grafon Urdu sentence-to-IPA corpus

Urdu sentences with aligned IPA-style transcriptions. **Annotation provenance is mixed**:
original corpus had human review, but specific masked symbols were later proposed by automatic
repair rules and a subset of additions used assistant-assisted QA. This dataset is not a
new independent linguistic gold standard. IPA includes punctuation, nasalization and
language-specific transcription conventions; consult the source corpus documentation.

The split has no exact sentence overlap (verified before upload) but substantial reusable
phrase templates and narrow lexical diversity. Evaluation metrics on these splits should
not be interpreted as broad Urdu generalization or SOTA. The 640 unadjudicated candidate
gold benchmark rows are excluded.

Columns: `text` (Urdu), `ipa` (space-aligned IPA-style representation).

Creator/contributor attribution and a redistributable dataset license must be added by
the rights holder before making this dataset public. Private staging is not a statement
of provenance verification.
"""


def main():
    ap=argparse.ArgumentParser()
    for key in ("train", "validation", "test"):
        ap.add_argument(f"--{key}", required=True)
    ap.add_argument("--repo", default="grafon-g2p/ur-data")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--rights-confirmed", action="store_true")
    ap.add_argument("--allow-existing", action="store_true")
    args=ap.parse_args()
    rows={n:read_tsv(getattr(args,n)) for n in ("train","validation","test")}
    sets={n:{s for s,_ in pairs} for n,pairs in rows.items()}
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        if sets[a]&sets[b]: raise SystemExit(f"Overlap between {a} and {b}")
    for name,pairs in rows.items():
        print(name,len(pairs),"sha256",sha256(getattr(args,name)))
    print("Hub target",args.repo,"private staging only")
    if not args.execute: return
    if not args.rights_confirmed:
        raise SystemExit("Dataset redistribution requires --rights-confirmed after checking consent/license")
    api=HfApi()
    who=api.whoami()
    org=args.repo.split('/')[0]
    if not any(isinstance(o,dict) and o.get('name')==org for o in who.get('orgs',[])):
        raise SystemExit(f"No membership in {org} verified for {who.get('name')}")
    try: existing=api.repo_info(args.repo,repo_type='dataset')
    except Exception: existing=None
    if existing is not None and (not args.allow_existing or not existing.private):
        raise SystemExit("Dataset repository exists or is public; manual review required")
    stage=Path('dist/hf-urdu-dataset')
    if stage.exists(): raise SystemExit(f"Staging folder already exists: {stage}")
    stage.mkdir(parents=True)
    (stage/'README.md').write_text(CARD,encoding='utf8')
    for name,pairs in rows.items():
        with (stage/f'{name}.tsv').open('w',encoding='utf8') as f:
            for text,ipa in pairs: f.write(f'{text}\t{ipa}\n')
    api.create_repo(args.repo,repo_type='dataset',private=True,exist_ok=args.allow_existing)
    api.upload_folder(repo_id=args.repo,repo_type='dataset',folder_path=str(stage),
                      commit_message='Stage Urdu G2P training corpus (mixed annotation provenance)')
    print('Uploaded private dataset:',args.repo)

if __name__=='__main__': main()
