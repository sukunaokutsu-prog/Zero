"""Gather the real training corpus (Project Gutenberg novels via GITenberg)
and the toy/curriculum corpus (TinyShakespeare)."""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from urllib.parse import quote_plus

TITLES = [
    "A Tale of Two Cities", "Dracula", "Jane Eyre", "Wuthering Heights",
    "The Picture of Dorian Gray", "Great Expectations", "Oliver Twist",
    "The Time Machine", "The War of the Worlds", "Treasure Island",
    "The Strange Case of Dr Jekyll and Mr Hyde",
    "Twenty Thousand Leagues under the Sea", "Around the World in Eighty Days",
    "Heart of Darkness", "Crime and Punishment", "The Adventures of Tom Sawyer",
    "Adventures of Huckleberry Finn", "The Count of Monte Cristo",
    "Les Miserables", "The Call of the Wild", "White Fang", "The Jungle Book",
    "The Odyssey", "Metamorphosis", "Anna Karenina", "War and Peace",
    "The Three Musketeers", "David Copperfield", "Sense and Sensibility",
    "Emma", "Gulliver's Travels", "The Iliad", "Don Quixote",
    "The Scarlet Letter",
]


def run(cmd, timeout=90):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def resolve(title, n=4):
    q = quote_plus(f"org:GITenberg {title}")
    r = run(["gh", "api", f"search/repositories?q={q}&per_page={n}",
             "--jq", ".items[].full_name"], timeout=30)
    if r is None or r.returncode != 0:
        return []
    return [l.strip() for l in r.stdout.splitlines() if l.strip()]


def strip_book(text):
    s, e = text.find("*** START OF"), text.find("*** END OF")
    return text[s:e].strip() if s != -1 and e != -1 and e > s else text.strip()


def extract(repo):
    with tempfile.TemporaryDirectory() as tmp:
        r = run(["git", "clone", "--depth", "1",
                 f"https://github.com/{repo}.git", f"{tmp}/b"], timeout=90)
        if r is None or r.returncode != 0:
            return None
        for root, _d, files in os.walk(f"{tmp}/b"):
            for fn in sorted(files):
                if fn.endswith(".txt"):
                    try:
                        t = open(os.path.join(root, fn), encoding="utf-8",
                                 errors="replace").read()
                        if "*** START OF" in t or len(t) > 20000:
                            return strip_book(t)
                    except Exception:
                        continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--max-mb", type=int, default=32)
    args = ap.parse_args()
    target = args.max_mb * 1024 * 1024

    # seed with any books already on disk
    parts = []
    books_dir = "data/books"
    if os.path.isdir(books_dir):
        for fn in sorted(os.listdir(books_dir)):
            if fn.endswith(".txt"):
                parts.append(strip_book(open(os.path.join(books_dir, fn),
                                            encoding="utf-8", errors="replace").read()))
    total = sum(len(p) for p in parts)
    print(f"seeded {len(parts)} books ({total/1e6:.2f} MB)")

    for title in TITLES:
        if total >= target:
            break
        for repo in resolve(title):
            text = extract(repo)
            if text:
                parts.append(text)
                total += len(text)
                print(f"  + {title:<38s} ({len(text)/1e6:.2f} MB)")
                break

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))
    print(f"corpus -> {args.out} ({total/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
