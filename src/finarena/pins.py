"""Hugging Face revisions pinned for reproducibility.

Each revision was the head of its repository on 2026-10-11, and none of them had changed since
before this project started (2026-10-01), so these are the versions behind the committed numbers.

To move a pin, change the hash here and rerun whatever depends on it.
"""

REVISIONS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "zeroshot/twitter-financial-news-sentiment": "ccbe24de388e287beb92dd393a335c376b350ac3",
    "Jean-Baptiste/financial_news_sentiment": "547fa98342d2c80734a1c841dbf3de2cfcaaab95",
}


def revision(repo_id: str) -> str | None:
    """Pinned revision for ``repo_id``; None (the library default) for a local path or an unlisted id."""
    return REVISIONS.get(repo_id)
