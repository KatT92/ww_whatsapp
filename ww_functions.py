"""
Utility functions for the WhatsApp Streamlit dashboard.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import pandas as pd


DEFAULT_WHATSAPP_PATTERN = r"(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}) - (.*?): (.*)"


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


def parse_whatsapp_text(
    text: str,
    source_name: str,
    pattern: str = DEFAULT_WHATSAPP_PATTERN,
) -> pd.DataFrame:
    """
    Parse WhatsApp exported text into a dataframe.

    Supports multiline messages by appending continuation lines to the previous
    valid message. Lines that do not match the pattern before any valid message
    are ignored.
    """
    data = []
    current = None
    compiled = re.compile(pattern)

    for raw_line in text.splitlines():
        line = raw_line.strip("\n")
        match = compiled.match(line)

        if match:
            if current is not None:
                data.append(current)

            date, time, player_name, message = match.groups()

            current = {
                "Date": date,
                "Time": time,
                "WA_Name": str(player_name).strip(),
                "Text": str(message).strip(),
                "Source": source_name,
            }
        else:
            if current is not None and line.strip():
                current["Text"] = f"{current['Text']}\n{line.strip()}"

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

    existing["Nickname"] = (
        existing["Nickname"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    mask = (
        existing["Nickname"].eq("")
        | existing["Nickname"].eq(existing["WA_Name"])
    )

    existing.loc[mask, "Nickname"] = (
        existing.loc[mask, "WA_Name"]
        .apply(default_nickname_from_wa_name)
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
    mapping.loc[blank_mask, "Nickname"] = (
        mapping.loc[blank_mask, "WA_Name"]
        .apply(default_nickname_from_wa_name)
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
        df.loc[missing_mask, "WA_Name"]
        .astype(str)
        .apply(default_nickname_from_wa_name)
    )
    return df


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
