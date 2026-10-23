"""Print source split statistics and disallow masks/cross-split sentence duplicates."""
import argparse
from grafon_urdu.text import read_tsv, vocabulary, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args()
    rows = {n: read_tsv(getattr(args, n)) for n in ("train", "validation", "test")}
    sentences = {n: {s for s, _ in data} for n, data in rows.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        intersection = sentences[a] & sentences[b]
        print(a, b, "exact sentence overlap:", len(intersection))
        if intersection:
            raise SystemExit("Split overlap; refusing release")
    for name, data in rows.items():
        print(name, "rows:", len(data), "unique sentences:", len(sentences[name]),
              "source words:", len({w for s,_ in data for w in s.split()}),
              "sha256:", sha256(getattr(args, name)))
        if len(data) != len(sentences[name]):
            raise SystemExit("Duplicate sentences; review split design")


if __name__ == "__main__": main()
