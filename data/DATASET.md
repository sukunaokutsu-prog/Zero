# Dataset provenance

All training text is **public domain in the United States**, gathered from
[Project Gutenberg](https://www.gutenberg.org/) via the [GITenberg](https://github.com/GITenberg)
GitHub mirror (direct web hosts are blocked in this environment).

## Corpora

| file | description | size |
|---|---|---|
| `data/toy.txt` | TinyShakespeare (Phase-A curriculum corpus) | 1.1 MB |
| `data/corpus.txt` | **144 public-domain books** — novels, essays, philosophy, history | **96.4 MB** |
| `data/mixed.tok.json` | byte-level BPE tokenizer (vocab 2048) | — |
| `data/mixed.ids.bin` | memory-mapped token ids (**32,781,784 tokens**) | 131 MB |

## Contents (144 books)

* **Novels:** Austen, the Brontës, Eliot, Gaskell, Dickens, Thackeray, Hardy,
  Doyle (all of Sherlock Holmes), Wells, Verne, Stevenson, Haggard, Burroughs,
  Hugo, Dumas, Balzac, Flaubert, Stendhal, Dostoevsky, Cooper, Stowe, Wallace,
  Alcott, Burnett, Barrie, Grahame, Spyri, Montgomery, Sewell, Orczy, Kipling,
  London, Twain, Hawthorne, Irving, Wharton, Chopin, Cather, Fitzgerald,
  Hemingway, Woolf, Forster, Joyce, Ford, Butler, Defoe, Bunyan, Voltaire,
  Fielding, Blackmore, Buchan, Childers, Morris, Bellamy, Abbott, Baum, James…
* **Essays / nonfiction / philosophy:** Thoreau, Emerson, Du Bois, Washington,
  Douglass, Northup, Smith, Darwin, Franklin, Paine, Hamilton/Madison/Jay,
  Machiavelli, More, Plato, Marcus Aurelius.

Every work is pre-1929 (or otherwise PD in the US); Gutenberg boilerplate is
stripped via the `*** START OF` / `*** END OF` markers. ~5 of the 144 books are
duplicated because the seed books were re-fetched in the expansion run (a
~3% duplication, noted for transparency).

## Reproducing

```bash
.venv/bin/python scripts/fetch_corpus.py --max-mb 110
.venv/bin/python scripts/prep_data.py --vocab 2048
```
