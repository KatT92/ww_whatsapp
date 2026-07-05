"""
Utility functions for the WhatsApp Streamlit dashboard.


"""

from __future__ import annotations

import re
from collections import Counter
from io import StringIO
from typing import Iterable, Optional

import emoji
import pandas as pd
from textblob import TextBlob


# =========================================================
# PARSING
# =========================================================

DEFAULT_WHATSAPP_PATTERN = (
    r"(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s-\s(.*?):\s(.*)"
)


def parse_whatsapp_text(
    text: str,
    source_name: str,
    pattern: str = DEFAULT_WHATSAPP_PATTERN,
) -> pd.DataFrame:
    """
    Parse WhatsApp exported text into a dataframe.

    Supports multi-line messages by appending continuation lines to the
    previous message.
    """
    rows: list[list[str]] = []
    current_row: Optional[list[str]] = None

    compiled = re.compile(pattern)

    for raw_line in text.splitlines():
        line = raw_line.strip("\ufeff")
        match = compiled.match(line)

        if match:
            if current_row is not None:
                rows.append(current_row)

            date, time, wa_name, message_text = match.groups()
            current_row = [date, time, wa_name, message_text, source_name]

        elif current_row is not None and line.strip():
            current_row[3] += "\n" + line.strip()

    if current_row is not None:
        rows.append(current_row)

    return pd.DataFrame(rows, columns=["Date", "Time", "WA_Name", "Text", "Source"])


def parse_whatsapp_file(
    file_obj,
    source_name: str,
    pattern: str = DEFAULT_WHATSAPP_PATTERN,
) -> pd.DataFrame:
    """
    Parse a Streamlit uploaded file, file path, or file-like object.
    """
    if isinstance(file_obj, str):
        with open(file_obj, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        content = file_obj.read()
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content)

    return parse_whatsapp_text(text, source_name=source_name, pattern=pattern)


def merge_whatsapp_chats(
    files: Iterable,
    source_mapping: dict[str, str],
    pattern: str = DEFAULT_WHATSAPP_PATTERN,
) -> pd.DataFrame:
    """
    Merge multiple WhatsApp chat exports.

    source_mapping should map each uploaded file name/path to the friendly chat name.
    """
    dfs = []

    for file_obj in files:
        file_key = getattr(file_obj, "name", str(file_obj))
        source_name = source_mapping.get(file_key, file_key)
        dfs.append(parse_whatsapp_file(file_obj, source_name, pattern))

    if not dfs:
        return pd.DataFrame(columns=["Date", "Time", "WA_Name", "Text", "Source"])

    return pd.concat(dfs, ignore_index=True)


# Backwards-compatible name from your earlier notebook.
def parse_whatsapp_messages(file_path, source_name, pattern=DEFAULT_WHATSAPP_PATTERN):
    return parse_whatsapp_file(file_path, source_name, pattern)


# =========================================================
# DATA PREP
# =========================================================


def get_sentiment(text: str) -> float:
    """Return TextBlob sentiment polarity for a message."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return float(TextBlob(text).sentiment.polarity)


def extract_emojis(text: str) -> list[str]:
    """Extract emoji characters from text."""
    if not isinstance(text, str):
        return []
    return [char for char in text if char in emoji.EMOJI_DATA]


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add DateTime, DateOnly, Hour, Sentiment, and Emoji columns.
    """
    df = df.copy()

    if df.empty:
        return df

    df["Text"] = df["Text"].fillna("").astype(str)
    df["WA_Name"] = df["WA_Name"].fillna("").astype(str)
    df["Source"] = df["Source"].fillna("").astype(str)

    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    df["DateOnly"] = df["DateTime"].dt.date
    df["Hour"] = df["DateTime"].dt.hour
    df["Sentiment"] = df["Text"].apply(get_sentiment)
    df["Emojis"] = df["Text"].apply(extract_emojis)

    if "Nickname" not in df.columns:
        df["Nickname"] = df["WA_Name"]

    return df


# =========================================================
# NICKNAME MAPPING FROM STREAMLIT DATA EDITOR
# =========================================================


def create_nickname_mapping(
    df: pd.DataFrame,
    existing_mapping: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create a mapping table for st.data_editor.

    Output columns:
    - WA_Name: original WhatsApp sender name
    - Nickname: replacement name used throughout the dashboard
    """
    if df.empty or "WA_Name" not in df.columns:
        return pd.DataFrame(columns=["WA_Name", "Nickname"])

    base = pd.DataFrame(
        {"WA_Name": sorted(df["WA_Name"].dropna().astype(str).unique())}
    )

    if existing_mapping is None or existing_mapping.empty:
        base["Nickname"] = base["WA_Name"]
        return base

    existing = existing_mapping.copy()

    if not {"WA_Name", "Nickname"}.issubset(existing.columns):
        base["Nickname"] = base["WA_Name"]
        return base

    existing["WA_Name"] = existing["WA_Name"].astype(str)
    existing["Nickname"] = existing["Nickname"].astype(str)
    existing = existing.drop_duplicates(subset=["WA_Name"], keep="last")

    merged = base.merge(existing, on="WA_Name", how="left")
    merged["Nickname"] = merged["Nickname"].fillna(merged["WA_Name"])

    return merged[["WA_Name", "Nickname"]]


def apply_nickname_mapping(df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace/create df['Nickname'] using the exact mapping from the data editor.
    No hidden alias logic is applied here.
    """
    df = df.copy()

    if df.empty:
        return df

    if mapping_df is None or mapping_df.empty:
        df["Nickname"] = df["WA_Name"]
        return df

    mapping = mapping_df.copy()

    if not {"WA_Name", "Nickname"}.issubset(mapping.columns):
        df["Nickname"] = df["WA_Name"]
        return df

    mapping["WA_Name"] = mapping["WA_Name"].astype(str)
    mapping["Nickname"] = mapping["Nickname"].astype(str).str.strip()
    mapping.loc[mapping["Nickname"].eq(""), "Nickname"] = mapping["WA_Name"]
    mapping = mapping.drop_duplicates(subset=["WA_Name"], keep="last")

    lookup = dict(zip(mapping["WA_Name"], mapping["Nickname"]))
    df["Nickname"] = df["WA_Name"].astype(str).map(lookup).fillna(df["WA_Name"])

    return df


# =========================================================
# TEXT/STATS HELPERS
# =========================================================


def count_word_mentions(df: pd.DataFrame, word: str, text_column: str = "Text") -> int:
    """Count whole-word occurrences across all messages."""
    if not word:
        return 0

    pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
    return int(
        df[text_column]
        .fillna("")
        .apply(lambda text: len(pattern.findall(str(text))))
        .sum()
    )


def word_mentions(df: pd.DataFrame, word: str) -> dict:
    """
    Count total mentions of a word and mentions by Nickname.
    Uses whole-word, case-insensitive matching.
    """
    if not word:
        return {"total_mentions": 0, "mentions_by_player": {}}

    pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
    mentions_by_player: Counter = Counter()
    total = 0

    for _, row in df.iterrows():
        text = str(row.get("Text", ""))
        nickname = row.get("Nickname", row.get("WA_Name", "Unknown"))
        count = len(pattern.findall(text))

        if count:
            mentions_by_player[nickname] += count
            total += count

    return {
        "total_mentions": int(total),
        "mentions_by_player": dict(mentions_by_player.most_common()),
    }


def count_cooccurrence(
    df: pd.DataFrame,
    word1: str,
    word2: str,
    text_column: str = "Text",
    whole_word: bool = True,
) -> int:
    """
    Count messages that contain both input words/phrases.
    For multi-word phrases, it searches the phrase exactly, case-insensitively.
    """
    if not word1 or not word2:
        return 0

    if whole_word:
        p1 = re.compile(rf"\b{re.escape(word1)}\b", flags=re.IGNORECASE)
        p2 = re.compile(rf"\b{re.escape(word2)}\b", flags=re.IGNORECASE)
    else:
        p1 = re.compile(re.escape(word1), flags=re.IGNORECASE)
        p2 = re.compile(re.escape(word2), flags=re.IGNORECASE)

    def has_both(text: str) -> bool:
        text = str(text)
        return bool(p1.search(text) and p2.search(text))

    return int(df[text_column].fillna("").apply(has_both).sum())


def count_phrase_mentions(
    df: pd.DataFrame, phrase: str, text_column: str = "Text"
) -> int:
    """Count exact phrase occurrences across all messages, case-insensitively."""
    if not phrase:
        return 0

    pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
    return int(
        df[text_column]
        .fillna("")
        .apply(lambda text: len(pattern.findall(str(text))))
        .sum()
    )


def count_word_by_player(df: pd.DataFrame, nickname: str, target_word: str) -> int:
    """Count whole-word mentions by one mapped Nickname."""
    if not nickname or not target_word or "Nickname" not in df.columns:
        return 0

    player_df = df[df["Nickname"].astype(str).str.lower() == nickname.lower()]
    return count_word_mentions(player_df, target_word)


def top_words(
    df: pd.DataFrame,
    text_column: str = "Text",
    n: int = 20,
    min_length: int = 2,
) -> pd.DataFrame:
    """Return a dataframe of the most common words."""
    text = " ".join(df[text_column].fillna("").astype(str)).lower()
    words = re.findall(r"\b[a-zA-Z']+\b", text)
    words = [w for w in words if len(w) >= min_length]
    counts = Counter(words).most_common(n)
    return pd.DataFrame(counts, columns=["Word", "Count"])


def top_emojis(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return a dataframe of the most common emojis."""
    all_emojis = []

    if "Emojis" in df.columns:
        for items in df["Emojis"]:
            if isinstance(items, list):
                all_emojis.extend(items)
    else:
        for text in df["Text"].fillna(""):
            all_emojis.extend(extract_emojis(str(text)))

    return pd.DataFrame(Counter(all_emojis).most_common(n), columns=["Emoji", "Count"])
