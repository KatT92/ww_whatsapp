from __future__ import annotations

import re
from collections import Counter
import pandas as pd

def top_words(
    df: pd.DataFrame,
    n: int = 30,
    text_column: str = "Text",
) -> pd.DataFrame:
    """
    Return top words from text_column.
    Use text_column='FilteredText' for stopword-filtered word-based graphs.
    """
    if df.empty or text_column not in df.columns:
        return pd.DataFrame(columns=["Word", "Count"])

    words = " ".join(df[text_column].fillna("").astype(str)).lower().split()

    return pd.DataFrame(
        Counter(words).most_common(n),
        columns=["Word", "Count"],
    )


def total_words_by_person(
    df: pd.DataFrame,
    text_column: str = "Text",
) -> pd.DataFrame:
    """
    Count total words by person.

    This should normally use original Text, not FilteredText.
    """
    if df.empty or text_column not in df.columns or "Nickname" not in df.columns:
        return pd.DataFrame(columns=["Person", "Total Words"])

    work = df.copy()
    work["_word_count"] = (
        work[text_column].fillna("").astype(str).str.split().apply(len)
    )

    out = (
        work.groupby("Nickname")["_word_count"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    out.columns = ["Person", "Total Words"]
    return out


def extract_emojis(text: str) -> list[str]:
    """Extract emoji characters from text."""
    if not isinstance(text, str):
        return []

    try:
        import emoji

        return [char for char in text if char in emoji.EMOJI_DATA]
    except Exception:
        emoji_pattern = re.compile(
            "["
            "\U0001f300-\U0001f5ff"
            "\U0001f600-\U0001f64f"
            "\U0001f680-\U0001f6ff"
            "\U0001f700-\U0001f77f"
            "\U0001f780-\U0001f7ff"
            "\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff"
            "\U0001fa00-\U0001faff"
            "\U00002700-\U000027bf"
            "\U00002600-\U000026ff"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.findall(text)


def top_emojis(
    df: pd.DataFrame,
    n: int = 30,
    text_column: str = "Text",
) -> pd.DataFrame:
    """
    Count top emojis.

    This should use original Text, not FilteredText.
    """
    if df.empty or text_column not in df.columns:
        return pd.DataFrame(columns=["Emoji", "Count"])

    counter = Counter()

    for text in df[text_column].fillna("").astype(str):
        counter.update(extract_emojis(text))

    return pd.DataFrame(
        counter.most_common(n),
        columns=["Emoji", "Count"],
    )


def count_cooccurrence(
    df: pd.DataFrame,
    word1: str,
    word2: str,
    text_column: str = "Text",
) -> int:
    """
    Count messages containing both words/phrases in text_column.
    """
    if df.empty or text_column not in df.columns:
        return 0

    word1 = str(word1).lower().strip()
    word2 = str(word2).lower().strip()

    if not word1 or not word2:
        return 0

    return int(
        df[text_column]
        .fillna("")
        .astype(str)
        .str.lower()
        .apply(lambda text: word1 in text and word2 in text)
        .sum()
    )


def word_mentions(
    df: pd.DataFrame,
    word: str,
    text_column: str = "Text",
) -> dict:
    """
    Count total mentions of a word and mentions by player.
    """
    if df.empty or text_column not in df.columns:
        return {
            "total_mentions": 0,
            "mentions_by_player": {},
        }

    word = str(word).lower().strip()

    if not word:
        return {
            "total_mentions": 0,
            "mentions_by_player": {},
        }

    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)

    mentions_by_player = Counter()
    total = 0

    for _, row in df.iterrows():
        text = str(row.get(text_column, ""))
        nickname = row.get("Nickname", "Unknown")

        count = len(pattern.findall(text))

        if count:
            mentions_by_player[nickname] += count
            total += count

    return {
        "total_mentions": total,
        "mentions_by_player": dict(mentions_by_player.most_common()),
    }


def count_word_by_player(
    df: pd.DataFrame,
    nickname: str,
    target_word: str,
    text_column: str = "Text",
) -> int:
    """
    Count how many times a specific player used a word.
    """
    if df.empty or text_column not in df.columns or "Nickname" not in df.columns:
        return 0

    nickname = str(nickname).lower().strip()
    target_word = str(target_word).lower().strip()

    if not nickname or not target_word:
        return 0

    player_df = df[df["Nickname"].astype(str).str.lower() == nickname]

    pattern = re.compile(rf"\b{re.escape(target_word)}\b", re.IGNORECASE)

    return int(
        player_df[text_column]
        .fillna("")
        .astype(str)
        .apply(lambda text: len(pattern.findall(text)))
        .sum()
    )


def count_phrase_mentions(
    df: pd.DataFrame,
    phrase: str,
    text_column: str = "Text",
) -> int:
    """
    Count exact phrase occurrences.
    """
    if df.empty or text_column not in df.columns:
        return 0

    phrase = str(phrase).lower().strip()

    if not phrase:
        return 0

    return int(
        df[text_column]
        .fillna("")
        .astype(str)
        .str.lower()
        .apply(lambda text: text.count(phrase))
        .sum()
    )
