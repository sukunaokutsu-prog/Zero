"""Gather a large real corpus: 150+ public-domain classics via GITenberg.

All works are public domain in the US (pre-1929, or otherwise PD), fetched from
Project Gutenberg's GITenberg GitHub mirror. Uses GitHub search per title with
a small delay to respect rate limits; failures are skipped gracefully.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from urllib.parse import quote_plus

TITLES = [
    # ---- novels (existing) ----
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
    # ---- Austen / Brontë / Eliot / Gaskell ----
    "Pride and Prejudice", "Frankenstein", "Moby Dick",
    "The Adventures of Sherlock Holmes", "Alice's Adventures in Wonderland",
    "Northanger Abbey", "Persuasion", "Mansfield Park",
    "Agnes Grey", "The Tenant of Wildfell Hall", "Shirley", "Villette",
    "The Professor", "Middlemarch", "Silas Marner", "The Mill on the Floss",
    "Cranford", "North and South", "Wives and Daughters",
    # ---- Dickens / Thackeray / Hardy ----
    "A Christmas Carol", "The Pickwick Papers", "Nicholas Nickleby",
    "Bleak House", "Hard Times", "Little Dorrit", "Dombey and Son",
    "Vanity Fair", "Tess of the d'Urbervilles", "Jude the Obscure",
    "Far from the Madding Crowd", "The Mayor of Casterbridge",
    # ---- Doyle / Wells / Verne / Stevenson / Haggard / Burroughs ----
    "A Study in Scarlet", "The Sign of the Four", "The Hound of the Baskervilles",
    "The Memoirs of Sherlock Holmes", "The Return of Sherlock Holmes",
    "The Valley of Fear", "The Lost World", "The Invisible Man",
    "The Island of Doctor Moreau", "The First Men in the Moon",
    "The Food of the Gods", "The Sleeper Awakes", "A Modern Utopia",
    "Journey to the Center of the Earth", "From the Earth to the Moon",
    "The Mysterious Island", "In Search of the Castaways",
    "Kidnapped", "The Master of Ballantrae", "King Solomon's Mines", "She",
    "Tarzan of the Apes", "A Princess of Mars", "The Prisoner of Zenda",
    # ---- Hugo / Dumas / Balzac / Flaubert / Stendhal / Dostoevsky ----
    "The Hunchback of Notre Dame", "The Man Who Laughs", "Twenty Years After",
    "The Vicomte of Bragelonne", "The Man in the Iron Mask", "The Black Tulip",
    "Father Goriot", "Eugenie Grandet", "Cousin Bette", "Madame Bovary",
    "The Red and the Black", "The Brothers Karamazov", "The Idiot",
    "Notes from Underground", "The Gambler",
    # ---- American canon ----
    "The Last of the Mohicans", "The Deerslayer", "Uncle Tom's Cabin",
    "Ben-Hur", "Quo Vadis", "Little Women", "Jo's Boys",
    "The Secret Garden", "A Little Princess", "Peter Pan",
    "The Wind in the Willows", "Heidi", "Anne of Green Gables",
    "Black Beauty", "The Scarlet Pimpernel", "Kim", "Captains Courageous",
    "The Sea-Wolf", "Martin Eden", "A Connecticut Yankee in King Arthur's Court",
    "The Prince and the Pauper", "Pudd'nhead Wilson",
    "The House of the Seven Gables", "The Legend of Sleepy Hollow",
    "Ethan Frome", "The Age of Innocence", "The House of Mirth", "The Awakening",
    "My Antonia", "O Pioneers", "The Great Gatsby", "The Sun Also Rises",
    "Mrs Dalloway", "To the Lighthouse", "A Passage to India", "Dubliners",
    "A Portrait of the Artist as a Young Man", "Ulysses", "The Good Soldier",
    "The Way of All Flesh", "Robinson Crusoe", "Moll Flanders",
    "The Pilgrim's Progress", "Candide", "Tom Jones", "Tristram Shandy",
    "Lorna Doone", "The Thirty-Nine Steps", "The Riddle of the Sands",
    "Erewhon", "News from Nowhere", "Looking Backward", "Flatland",
    "The King in Yellow", "The House on the Borderland", "The Great God Pan",
    "Carmilla", "The Beetle", "Vathek", "The Castle of Otranto", "The Monk",
    "Melmoth the Wanderer", "The Last Man", "The Railway Children",
    "Five Children and It", "The Story of the Amulet",
    "The Phoenix and the Carpet", "The Wonderful Wizard of Oz",
    "The Portrait of a Lady", "Washington Square", "Daisy Miller",
    "The Turn of the Screw", "The Ambassadors",
    # ---- essays / nonfiction / philosophy (public-domain translations) ----
    "Walden", "Civil Disobedience", "Self-Reliance", "Nature",
    "The Souls of Black Folk", "Up From Slavery",
    "The Narrative of the Life of Frederick Douglass", "Twelve Years a Slave",
    "The Wealth of Nations", "On the Origin of Species", "The Voyage of the Beagle",
    "The Descent of Man", "The Autobiography of Benjamin Franklin",
    "Common Sense", "The Rights of Man", "The Federalist Papers",
    "The Prince", "Utopia", "The Republic", "Nicomachean Ethics",
    "The Meditations of Marcus Aurelius", "History of the Peloponnesian War",
    "The Aeneid", "Paradise Lost", "The Divine Comedy", "Faust",
    "Leaves of Grass", "The Waste Land", "Spoon River Anthology",
    "The Song of Hiawatha", "The Raven", "The Complete Works of William Shakespeare",
    "The Complete Poetical Works of Edgar Allan Poe",
]


def run(cmd, timeout=90):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def resolve(title, n=3):
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
    ap.add_argument("--max-mb", type=int, default=110)
    args = ap.parse_args()
    target = args.max_mb * 1024 * 1024

    seen = set()
    parts = []
    books_dir = "data/books"
    if os.path.isdir(books_dir):
        for fn in sorted(os.listdir(books_dir)):
            if fn.endswith(".txt"):
                parts.append(strip_book(open(os.path.join(books_dir, fn),
                                            encoding="utf-8", errors="replace").read()))
    total = sum(len(p) for p in parts)
    print(f"seeded {len(parts)} books ({total/1e6:.2f} MB)", flush=True)

    n_ok = n_skip = 0
    for title in TITLES:
        if total >= target:
            break
        for repo in resolve(title):
            if repo in seen:
                continue
            seen.add(repo)
            text = extract(repo)
            if text:
                parts.append(text)
                total += len(text)
                n_ok += 1
                print(f"  + {title:<42s} ({len(text)/1e6:.2f} MB) [total {total/1e6:.1f} MB]", flush=True)
                break
        else:
            n_skip += 1
        time.sleep(0.4)  # respect search API rate limits

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))
    print(f"done: +{n_ok} books ({n_skip} skipped) -> {args.out} ({total/1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
