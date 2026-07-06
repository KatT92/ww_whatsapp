"""
Utility functions for the WhatsApp Streamlit dashboard.
"""

from __future__ import annotations

import re
from typing import Optional
import pandas as pd


DEFAULT_WHATSAPP_PATTERN = r"(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}) - (.*?): (.*)"

BRACKETED_WHATSAPP_PATTERN = r"\[(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}:\d{2})\] (.*?): (.*)"

WHATSAPP_PATTERNS = {
    "Pattern 1 - Default WhatsApp": DEFAULT_WHATSAPP_PATTERN,
    "Pattern 2 - Bracketed WhatsApp with seconds": BRACKETED_WHATSAPP_PATTERN,
    "Custom pattern": None,
}

DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "so",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "im",
    "i'm",
    "ive",
    "i've",
    "dont",
    "don't",
    "didnt",
    "didn't",
    "cant",
    "can't",
    "wont",
    "won't",
    "would",
    "could",
    "should",
    "like",
    "just",
    "really",
    "get",
    "got",
    "one",
    "also",
    "yeah",
    "yes",
    "no",
    "not",
}

SYSTEM_MESSAGE_PHRASES = [
    "added you to",
    "joined from the community",
    "created group",
    "created this group",
    "changed the subject",
    "changed this group's icon",
    "changed the group description",
    "joined using",
    "invite link",
    "messages and calls are end-to-end encrypted",
    "welcome to the group",
    "you were added",
    "you joined",
    "removed",
    "was removed",
    "left",
]

def is_system_message(*parts: str) -> bool:
    """
    Returns True if any supplied text contains a WhatsApp system message.
    """
    text = " ".join(str(p).lower() for p in parts)

    return any(
        phrase in text
        for phrase in SYSTEM_MESSAGE_PHRASES
    )

def parse_whatsapp_text(
    text: str,
    source_name: str,
    pattern: str = DEFAULT_WHATSAPP_PATTERN,
) -> pd.DataFrame:
    """
    Parse WhatsApp exported text into a dataframe.

    Supports multiline messages by appending continuation lines to the previous
    valid message.

    Filters out WhatsApp system messages
    """
    data = []
    current = None
    compiled = re.compile(pattern)

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if is_system_message(line):
            continue

        match = compiled.match(line)

        if match:
            if current is not None:
                data.append(current)

            date, time, player_name, message = match.groups()

            player_name = str(player_name).strip()
            message = str(message).strip()

            if is_system_message(player_name, message):
                current = None
                continue

            current = {
                "Date": date,
                "Time": time,
                "WA_Name": player_name,
                "Text": message,
                "Source": source_name,
            }

        elif current is not None and line:
            current["Text"] += f"\n{line}"

    if current is not None:
        data.append(current)

    return pd.DataFrame(
        data,
        columns=["Date", "Time", "WA_Name", "Text", "Source"],
    )


def clean_text_for_words(text: str, remove_stopwords: bool = True) -> str:
    """
    Lowercase text, keep alphabetic words/apostrophes, and optionally remove stopwords.
    """
    if not isinstance(text, str):
        return ""

    words = re.findall(r"[A-Za-z']+", text.lower())

    if remove_stopwords:
        words = [w for w in words if w not in DEFAULT_STOPWORDS and len(w) > 1]

    return " ".join(words)


def get_sentiment(text: str) -> float:
    """
    Sentiment helper.

    Uses TextBlob if installed. Falls back to 0.0 if TextBlob is unavailable.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    try:
        from textblob import TextBlob

        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add parsed datetime, hour, date, sentiment, and FilteredText columns.
    """
    df = df.copy()

    for col in ["Date", "Time", "WA_Name", "Text", "Source"]:
        if col not in df.columns:
            df[col] = ""

    if df.empty:
        df["DateTime"] = pd.to_datetime(pd.Series([], dtype="object"))
        df["Hour"] = pd.Series([], dtype="float")
        df["DateOnly"] = pd.Series([], dtype="object")
        df["FilteredText"] = pd.Series([], dtype="object")
        df["Sentiment"] = pd.Series([], dtype="float")
        return df

    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce",
        dayfirst=True,
    )

    df["Hour"] = df["DateTime"].dt.hour
    df["DateOnly"] = df["DateTime"].dt.date

    df["Text"] = df["Text"].fillna("").astype(str)
    df["FilteredText"] = df["Text"].apply(
        lambda x: clean_text_for_words(x, remove_stopwords=True)
    )

    df["Sentiment"] = df["Text"].apply(get_sentiment)

    return df


def default_nickname_from_wa_name(name: str) -> str:
    """
    Default nickname before user edits:
    - '~ First Last' -> 'First'
    - 'First Last' -> 'First'
    - '+447...' stays as phone number
    """
    if not isinstance(name, str):
        return ""

    name = name.strip()

    if name.startswith("+"):
        return name

    name = name.lstrip("~").strip()

    if not name:
        return ""

    return name.split()[0]


def create_nickname_mapping(
    df: pd.DataFrame,
    existing_mapping: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create a mapping table for st.data_editor.

    Output columns:
    - Ignore: tick True to remove this WA_Name from analysis
    - WA_Name: original WhatsApp sender name
    - Nickname: replacement name used throughout the dashboard
    """
    columns = ["Ignore", "WA_Name", "Nickname"]

    if df.empty or "WA_Name" not in df.columns:
        return pd.DataFrame(columns=columns)

    base = pd.DataFrame(
        {"WA_Name": sorted(df["WA_Name"].dropna().astype(str).unique())}
    )

    if existing_mapping is None or existing_mapping.empty:
        base["Nickname"] = base["WA_Name"].apply(default_nickname_from_wa_name)
        base["Ignore"] = False
        return base[columns]

    existing = existing_mapping.copy()

    if not {"WA_Name", "Nickname"}.issubset(existing.columns):
        base["Nickname"] = base["WA_Name"].apply(default_nickname_from_wa_name)
        base["Ignore"] = False
        return base[columns]

    if "Ignore" not in existing.columns:
        existing["Ignore"] = False

    existing["WA_Name"] = existing["WA_Name"].astype(str)

    existing["Nickname"] = existing["Nickname"].fillna("").astype(str).str.strip()

    mask = existing["Nickname"].eq("") | existing["Nickname"].eq(existing["WA_Name"])

    existing.loc[mask, "Nickname"] = existing.loc[mask, "WA_Name"].apply(
        default_nickname_from_wa_name
    )

    existing["Ignore"] = existing["Ignore"].fillna(False).astype(bool)

    existing = existing.drop_duplicates(subset=["WA_Name"], keep="last")

    merged = base.merge(
        existing[["WA_Name", "Nickname", "Ignore"]],
        on="WA_Name",
        how="left",
    )

    merged["Nickname"] = merged["Nickname"].fillna(
        merged["WA_Name"].apply(default_nickname_from_wa_name)
    )
    merged["Ignore"] = merged["Ignore"].fillna(False).astype(bool)

    return merged[columns]


def apply_nickname_mapping(
    df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    remove_ignored: bool = True,
) -> pd.DataFrame:
    """
    Replace/create df['Nickname'] using the exact mapping from the data editor.

    If mapping_df has an Ignore column and remove_ignored=True,
    rows with ignored WA_Name values are removed from the returned dataframe.

    No hidden alias logic is applied here.
    """
    df = df.copy()

    if df.empty:
        return df

    if "WA_Name" not in df.columns:
        return df

    if mapping_df is None or mapping_df.empty:
        df["Nickname"] = df["WA_Name"].astype(str).apply(default_nickname_from_wa_name)
        return df

    mapping = mapping_df.copy()

    if not {"WA_Name", "Nickname"}.issubset(mapping.columns):
        df["Nickname"] = df["WA_Name"].astype(str).apply(default_nickname_from_wa_name)
        return df

    if "Ignore" not in mapping.columns:
        mapping["Ignore"] = False

    mapping["WA_Name"] = mapping["WA_Name"].astype(str)
    mapping["Nickname"] = mapping["Nickname"].astype(str).str.strip()
    blank_mask = mapping["Nickname"].eq("")
    mapping.loc[blank_mask, "Nickname"] = mapping.loc[blank_mask, "WA_Name"].apply(
        default_nickname_from_wa_name
    )
    mapping["Ignore"] = mapping["Ignore"].fillna(False).astype(bool)

    mapping = mapping.drop_duplicates(subset=["WA_Name"], keep="last")

    if remove_ignored:
        ignored_names = mapping.loc[mapping["Ignore"], "WA_Name"].tolist()
        df = df[~df["WA_Name"].astype(str).isin(ignored_names)].copy()

    lookup = dict(zip(mapping["WA_Name"], mapping["Nickname"]))

    df["Nickname"] = df["WA_Name"].astype(str).map(lookup)

    missing_mask = df["Nickname"].isna() | df["Nickname"].astype(str).str.strip().eq("")

    df.loc[missing_mask, "Nickname"] = (
        df.loc[missing_mask, "WA_Name"].astype(str).apply(default_nickname_from_wa_name)
    )
    return df

