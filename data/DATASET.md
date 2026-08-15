# Dataset provenance

All training text in this repository is **public domain in the United States**
and was gathered from the [Project Gutenberg](https://www.gutenberg.org/)
collection via the [GITenberg](https://github.com/GITenberg) GitHub mirror
(because direct web hosts are blocked in the training environment).

## Corpora

| file | description | size |
|---|---|---|
| `data/input.txt` | TinyShakespeare (classic char-level LM benchmark) | 1.1 MB |
| `data/gutenberg.txt` | 5 novels (early small corpus) | 3.0 MB |
| `data/corpus.txt` | **36 novels** (the "beast" corpus) | 31.6 MB |
| `data/corpus.tok.json` | byte-level BPE tokenizer (vocab 2048) | — |
| `data/corpus.ids.bin` | memory-mapped token ids (10,654,151 tokens) | 42.6 MB |

## The 36 books in `data/corpus.txt`

1. The Adventures of Sherlock Holmes — Doyle (1661)
2. Pride and Prejudice — Austen (1342)
3. Frankenstein — Shelley (84)
4. Alice's Adventures in Wonderland — Carroll (11)
5. Moby Dick; or, The Whale — Melville (2701)
6. A Tale of Two Cities — Dickens (98)
7. Dracula — Stoker (345)
8. Jane Eyre — Brontë (1260)
9. Wuthering Heights — Brontë (768)
10. The Picture of Dorian Gray — Wilde (174)
11. Great Expectations — Dickens (1400)
12. Oliver Twist — Dickens (730)
13. The Time Machine — Wells (35)
14. The War of the Worlds — Wells (36)
15. Treasure Island — Stevenson (120)
16. The Strange Case of Dr. Jekyll and Mr. Hyde — Stevenson (42)
17. Twenty Thousand Leagues under the Sea — Verne (164)
18. Around the World in Eighty Days — Verne (2154)
19. Heart of Darkness — Conrad (219)
20. Crime and Punishment — Dostoevsky (2554)
21. The Adventures of Tom Sawyer — Twain (74)
22. Adventures of Huckleberry Finn — Twain (76)
23. The Count of Monte Cristo — Dumas (1184)
24. Les Misérables — Hugo (135)
25. The Call of the Wild — London (215)
26. White Fang — London (910)
27. The Jungle Book — Kipling (140)
28. The Odyssey — Homer (3160)
29. Metamorphosis — Kafka (5200)
30. Anna Karenina — Tolstoy (1399)
31. War and Peace — Tolstoy (2600)
32. The Three Musketeers — Dumas (1257)
33. David Copperfield — Dickens (766)
34. Sense and Sensibility — Austen (161)
35. Emma — Austen (158)
36. Gulliver's Travels — Swift (829)

Numbers in parentheses are Project Gutenberg ebook IDs. Gutenberg boilerplate
(headers/footers) was stripped via the `*** START OF` / `*** END OF` markers.
Books are concatenated with blank-line separators; the corpus is then
tokenized (byte-level BPE) and stored as a memory-mapped int32 file.

## Reproducing

```bash
bash scripts/setup.sh              # venv + CPU torch + tokenizers
.venv/bin/python scripts/fetch_corpus.py --max-mb 30   # downloads the books
.venv/bin/python scripts/prep_corpus.py --vocab 2048    # BPE + mmap token file
```

## Notes on scale and licensing

* These 36 novels are high-quality, human-written, long-form text — good
  training signal for a small LM — but they are ~5 orders of magnitude smaller
  than web-scale corpora (Common Crawl / OpenWebText / FineWeb are tens of TB
  and are unreachable from this environment).
* All works are public domain in the US; re-check your local jurisdiction for
  re-distribution rules (some translations have their own copyright status).
