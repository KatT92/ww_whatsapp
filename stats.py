from __future__ import annotations

import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


def small_fig(width: int = 6, height: int = 3):
    """Create a compact matplotlib figure."""
    return plt.subplots(figsize=(width, height))


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


def longest_messages(
    df: pd.DataFrame,
    n: int = 10,
    text_column: str = "Text",
) -> pd.DataFrame:
    if df.empty or text_column not in df.columns:
        return pd.DataFrame(columns=["Nickname", "Text", "Word Count", "Source"])

    out = df.copy()
    out["Word Count"] = out[text_column].fillna("").astype(str).str.split().apply(len)

    return (
        out[["Nickname", text_column, "Word Count", "Source"]]
        .sort_values("Word Count", ascending=False)
        .head(n)
        .rename(columns={text_column: "Text"})
        .reset_index(drop=True)
    )


def ngrams(
    df: pd.DataFrame,
    ngram_size: int = 2,
    top_n: int = 20,
    text_column: str = "FilteredText",
) -> pd.DataFrame:
    if df.empty or text_column not in df.columns:
        return pd.DataFrame(columns=["Phrase", "Count"])

    words = " ".join(df[text_column].fillna("").astype(str)).lower().split()

    if len(words) < ngram_size:
        return pd.DataFrame(columns=["Phrase", "Count"])

    phrases = zip(*[words[i:] for i in range(ngram_size)])
    counts = Counter(" ".join(p) for p in phrases)

    return pd.DataFrame(
        counts.most_common(top_n),
        columns=["Phrase", "Count"],
    )


def general_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Stat", "Value"])

    total_words = df["Text"].fillna("").astype(str).str.split().apply(len).sum()

    avg_words = df["Text"].fillna("").astype(str).str.split().apply(len).mean()

    stats = {
        "Messages": len(df),
        "People": df["Nickname"].nunique(),
        "Chats": df["Source"].nunique(),
        "Days": df["DateOnly"].nunique(),
        "Total words": int(total_words),
        "Average words per message": round(avg_words, 2),
        "Average sentiment": round(df["Sentiment"].mean(), 3),
    }

    return pd.DataFrame(stats.items(), columns=["Stat", "Value"])


def most_active_time_by_player(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"Nickname", "Hour"}.issubset(df.columns):
        return pd.DataFrame(columns=["Nickname", "Most Active Hour", "Messages"])

    counts = (
        df.groupby(["Nickname", "Hour"])
        .size()
        .reset_index(name="Messages")
        .sort_values(["Nickname", "Messages"], ascending=[True, False])
    )

    return (
        counts.drop_duplicates("Nickname")
        .rename(columns={"Hour": "Most Active Hour"})
        .sort_values("Messages", ascending=False)
        .reset_index(drop=True)
    )


def build_alias_lookup(mapping_df: pd.DataFrame) -> dict:
    """
    Uses the data editor mapping as the source of truth.
    WA_Name and Nickname can both be detected as mention terms.
    """
    lookup = {}

    if mapping_df is None or mapping_df.empty:
        return lookup

    for _, row in mapping_df.iterrows():
        if bool(row.get("Ignore", False)):
            continue

        nickname = str(row.get("Nickname", "")).strip()
        wa_name = str(row.get("WA_Name", "")).strip()

        if nickname:
            lookup[nickname.lower()] = nickname

        if wa_name:
            lookup[wa_name.lower()] = nickname or wa_name

    return lookup


def extract_mentions(text: str, alias_lookup: dict) -> list[str]:
    if not isinstance(text, str):
        return []

    text_lower = text.lower()
    mentions = []

    for alias, canonical in alias_lookup.items():
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, text_lower):
            mentions.append(canonical)

    return mentions


def mention_matrix(
    df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    normalise: bool = False,
) -> pd.DataFrame:
    if df.empty or not {"Nickname", "Text"}.issubset(df.columns):
        return pd.DataFrame()

    alias_lookup = build_alias_lookup(mapping_df)
    players = sorted(df["Nickname"].dropna().astype(str).unique())

    matrix = pd.DataFrame(0, index=players, columns=players, dtype=float)

    for _, row in df.iterrows():
        speaker = str(row["Nickname"])
        text = str(row["Text"])

        mentions = extract_mentions(text, alias_lookup)

        for mentioned in mentions:
            if mentioned in matrix.columns and speaker in matrix.index:
                if mentioned != speaker:
                    matrix.loc[speaker, mentioned] += 1

    if normalise:
        row_sums = matrix.sum(axis=1)
        matrix = matrix.div(row_sums.replace(0, 1), axis=0)

    return matrix.reset_index().rename(columns={"index": "Speaker"})


def mentions_received(
    df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    matrix = mention_matrix(df, mapping_df, normalise=False)

    if matrix.empty:
        return pd.DataFrame(columns=["Player", "Mentions Received"])

    values = (
        matrix.drop(columns=["Speaker"])
        .sum(axis=0)
        .sort_values(ascending=False)
        .reset_index()
    )

    values.columns = ["Player", "Mentions Received"]
    return values


def mentions_made(
    df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    matrix = mention_matrix(df, mapping_df, normalise=False)

    if matrix.empty:
        return pd.DataFrame(columns=["Speaker", "Mentions Made"])

    out = matrix.copy()
    out["Mentions Made"] = out.drop(columns=["Speaker"]).sum(axis=1)

    return out[["Speaker", "Mentions Made"]].sort_values(
        "Mentions Made",
        ascending=False,
    )


def messages_between_players(
    df: pd.DataFrame,
    player_a: str,
    player_b: str,
    text_column: str = "Text",
) -> pd.DataFrame:
    rows = []

    for sender, target in [(player_a, player_b), (player_b, player_a)]:
        sender_df = df[df["Nickname"].astype(str) == sender].copy()

        mentions_target = (
            sender_df[text_column]
            .fillna("")
            .astype(str)
            .str.contains(
                rf"\b{re.escape(target)}\b",
                case=False,
                regex=True,
            )
        )

        subset = sender_df[mentions_target]

        rows.append(
            {
                "From": sender,
                "To": target,
                "Messages mentioning target": len(subset),
                "Average sentiment": round(subset["Sentiment"].mean(), 3)
                if len(subset)
                else 0,
            }
        )

    return pd.DataFrame(rows)


def build_mention_terms_from_mapping(mapping_df: pd.DataFrame) -> dict:
    """
    Only uses WA_Name and Nickname from the data editor.
    """
    terms = {}

    if mapping_df is None or mapping_df.empty:
        return terms

    for _, row in mapping_df.iterrows():
        if bool(row.get("Ignore", False)):
            continue

        wa_name = str(row.get("WA_Name", "")).strip()
        nickname = str(row.get("Nickname", "")).strip()
        canonical = nickname or wa_name

        if wa_name:
            terms.setdefault(canonical, set()).add(wa_name.lower())

        if nickname:
            terms.setdefault(canonical, set()).add(nickname.lower())

    return terms


def text_mentions_player(text: str, player: str, mention_terms: dict) -> bool:
    if not isinstance(text, str):
        return False

    text_lower = text.lower()

    terms = mention_terms.get(player, {str(player).lower()})

    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", text_lower):
            return True

    return False


def sentiment_from_player_to_player_by_day(
    df: pd.DataFrame,
    from_player: str,
    to_player: str,
    mapping_df: pd.DataFrame,
    text_column: str = "Text",
) -> pd.DataFrame:
    """
    Daily average sentiment in messages from from_player that mention to_player.
    """
    if df.empty:
        return pd.DataFrame(columns=["DateOnly", "Average Sentiment", "Messages"])

    mention_terms = build_mention_terms_from_mapping(mapping_df)

    sender_df = df[df["Nickname"].astype(str) == from_player].copy()

    if sender_df.empty:
        return pd.DataFrame(columns=["DateOnly", "Average Sentiment", "Messages"])

    mask = (
        sender_df[text_column]
        .fillna("")
        .astype(str)
        .apply(lambda text: text_mentions_player(text, to_player, mention_terms))
    )

    target_df = sender_df[mask].copy()

    if target_df.empty:
        return pd.DataFrame(columns=["DateOnly", "Average Sentiment", "Messages"])

    out = (
        target_df.groupby("DateOnly")
        .agg(
            **{
                "Average Sentiment": ("Sentiment", "mean"),
                "Messages": ("Text", "count"),
            }
        )
        .reset_index()
    )

    out["Average Sentiment"] = out["Average Sentiment"].round(3)

    return out


def sentiment_about_player_from_all(
    df: pd.DataFrame,
    target_player: str,
    mapping_df: pd.DataFrame,
    text_column: str = "Text",
) -> pd.DataFrame:
    """
    Average sentiment from every sender in messages mentioning target_player.
    """
    if df.empty:
        return pd.DataFrame(columns=["From", "Average Sentiment", "Messages"])

    mention_terms = build_mention_terms_from_mapping(mapping_df)

    work = df.copy()

    work = work[work["Nickname"].astype(str) != target_player]

    mask = (
        work[text_column]
        .fillna("")
        .astype(str)
        .apply(lambda text: text_mentions_player(text, target_player, mention_terms))
    )

    work = work[mask]

    if work.empty:
        return pd.DataFrame(columns=["From", "Average Sentiment", "Messages"])

    out = (
        work.groupby("Nickname")
        .agg(
            **{
                "Average Sentiment": ("Sentiment", "mean"),
                "Messages": ("Text", "count"),
            }
        )
        .reset_index()
        .rename(columns={"Nickname": "From"})
        .sort_values("Average Sentiment", ascending=False)
    )

    out["Average Sentiment"] = out["Average Sentiment"].round(3)

    return out


def sentiment_from_player_to_all(
    df: pd.DataFrame,
    from_player: str,
    mapping_df: pd.DataFrame,
    text_column: str = "Text",
) -> pd.DataFrame:
    """
    Average sentiment from one sender towards every mentioned player.
    """
    if df.empty:
        return pd.DataFrame(columns=["To", "Average Sentiment", "Messages"])

    mention_terms = build_mention_terms_from_mapping(mapping_df)

    sender_df = df[df["Nickname"].astype(str) == from_player].copy()

    if sender_df.empty:
        return pd.DataFrame(columns=["To", "Average Sentiment", "Messages"])

    rows = []

    for target_player in sorted(df["Nickname"].dropna().astype(str).unique()):
        if target_player == from_player:
            continue

        mask = (
            sender_df[text_column]
            .fillna("")
            .astype(str)
            .apply(
                lambda text: text_mentions_player(text, target_player, mention_terms)
            )
        )

        target_df = sender_df[mask]

        if target_df.empty:
            continue

        rows.append(
            {
                "To": target_player,
                "Average Sentiment": round(target_df["Sentiment"].mean(), 3),
                "Messages": len(target_df),
            }
        )

    return (
        pd.DataFrame(rows).sort_values("Average Sentiment", ascending=False)
        if rows
        else pd.DataFrame(columns=["To", "Average Sentiment", "Messages"])
    )


def messages_between_players_by_day(
    df: pd.DataFrame,
    player_a: str,
    player_b: str,
    mapping_df: pd.DataFrame,
    text_column: str = "Text",
) -> pd.DataFrame:
    """
    Daily message counts where one selected player mentions the other.

    Returns:
        DateOnly | Direction | Messages
    """

    if df.empty:
        return pd.DataFrame(columns=["DateOnly", "Direction", "Messages"])

    mention_terms = build_mention_terms_from_mapping(mapping_df)

    rows = []

    for sender, target in [(player_a, player_b), (player_b, player_a)]:
        sender_df = df[df["Nickname"] == sender].copy()

        if sender_df.empty:
            continue

        mask = (
            sender_df[text_column]
            .fillna("")
            .astype(str)
            .apply(
                lambda text: text_mentions_player(
                    text,
                    target,
                    mention_terms,
                )
            )
        )

        counts = sender_df[mask].groupby("DateOnly").size().reset_index(name="Messages")

        counts["Direction"] = f"{sender} → {target}"

        rows.append(counts)

    if not rows:
        return pd.DataFrame(columns=["DateOnly", "Direction", "Messages"])

    return pd.concat(rows, ignore_index=True)


def most_active_hour_all_players(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"Nickname", "Hour"}.issubset(df.columns):
        return pd.DataFrame(columns=["Nickname", "Most Active Hour", "Messages"])

    counts = (
        df.groupby(["Nickname", "Hour"])
        .size()
        .reset_index(name="Messages")
        .sort_values(["Nickname", "Messages"], ascending=[True, False])
    )

    return (
        counts.drop_duplicates("Nickname")
        .rename(columns={"Hour": "Most Active Hour"})
        .sort_values("Messages", ascending=False)
        .reset_index(drop=True)
    )


def most_active_day_all_players(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"Nickname", "DateOnly"}.issubset(df.columns):
        return pd.DataFrame(columns=["Nickname", "Most Active Day", "Messages"])

    counts = (
        df.groupby(["Nickname", "DateOnly"])
        .size()
        .reset_index(name="Messages")
        .sort_values(["Nickname", "Messages"], ascending=[True, False])
    )

    return (
        counts.drop_duplicates("Nickname")
        .rename(columns={"DateOnly": "Most Active Day"})
        .sort_values("Messages", ascending=False)
        .reset_index(drop=True)
    )


def plot_normalised_mention_heatmap(
    df: pd.DataFrame,
    mapping_df: pd.DataFrame,
):
    matrix = mention_matrix(
        df,
        mapping_df,
        normalise=True,
    )

    fig, ax = small_fig(12, 8)

    if matrix.empty:
        return fig, ax, matrix

    heatmap_data = matrix.set_index("Speaker")

    im = ax.imshow(heatmap_data.values, aspect="auto")

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(
        heatmap_data.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            ax.text(
                j,
                i,
                f"{heatmap_data.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_title("Normalised mentions between players")
    ax.set_xlabel("Mentioned player")
    ax.set_ylabel("Speaker")

    fig.colorbar(im, ax=ax, label="Normalised mentions")
    fig.tight_layout()

    return fig, ax, matrix


def plot_most_active_hour_by_player(
    df: pd.DataFrame,
    min_messages: int = 5,
):
    """
    Scatter plot showing each player's most active hour.

    Bubble size = number of messages during that hour.
    """

    if df.empty or not {"Nickname", "Hour"}.issubset(df.columns):
        fig, ax = small_fig(9, 5)
        return fig, ax, pd.DataFrame()

    work = df.copy()

    work["Hour"] = pd.to_numeric(work["Hour"], errors="coerce")
    work = work.dropna(subset=["Hour"])
    work["Hour"] = work["Hour"].astype(int)

    # Ignore players with very few messages
    work = work.groupby("Nickname").filter(lambda x: len(x) >= min_messages)

    player_hour_activity = (
        work.groupby(["Nickname", "Hour"]).size().reset_index(name="Messages")
    )

    if player_hour_activity.empty:
        fig, ax = small_fig(9, 5)
        return fig, ax, pd.DataFrame()

    most_active = player_hour_activity.loc[
        player_hour_activity.groupby("Nickname")["Messages"].idxmax()
    ].sort_values(["Hour", "Nickname"])

    fig, ax = small_fig(10, 6)

    for _, row in most_active.iterrows():
        ax.scatter(
            row["Hour"],
            row["Messages"],
            s=max(row["Messages"] * 10, 40),
            alpha=0.7,
        )

        ax.text(
            row["Hour"],
            row["Messages"],
            f"{row['Nickname']} ({row['Messages']})",
            fontsize=9,
            ha="left",
            va="center",
        )

    ax.set_title("Most active hour for each player")
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Number of messages")
    ax.set_xticks(range(24))
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    return fig, ax, most_active


def plot_most_active_day_by_player(
    df: pd.DataFrame,
    min_messages: int = 5,
):
    """
    Scatter plot showing each player's most active day.

    Bubble size = number of messages on that day.
    """

    if df.empty or not {"Nickname", "DateOnly"}.issubset(df.columns):
        fig, ax = small_fig(9, 5)
        return fig, ax, pd.DataFrame()

    work = df.copy()

    work = work.groupby("Nickname").filter(lambda x: len(x) >= min_messages)

    player_day_activity = (
        work.groupby(["Nickname", "DateOnly"]).size().reset_index(name="Messages")
    )

    if player_day_activity.empty:
        fig, ax = small_fig(9, 5)
        return fig, ax, pd.DataFrame()

    most_active = player_day_activity.loc[
        player_day_activity.groupby("Nickname")["Messages"].idxmax()
    ].sort_values(["DateOnly", "Nickname"])

    fig, ax = small_fig(10, 6)

    for _, row in most_active.iterrows():
        ax.scatter(
            row["DateOnly"],
            row["Messages"],
            s=max(row["Messages"] * 10, 40),
            alpha=0.7,
        )

        ax.text(
            row["DateOnly"],
            row["Messages"],
            f"{row['Nickname']} ({row['Messages']})",
            fontsize=9,
            ha="left",
            va="center",
        )

    ax.set_title("Most active day for each player")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of messages")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    return fig, ax, most_active


def cumulative_messages_over_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"Nickname", "DateOnly"}.issubset(df.columns):
        return pd.DataFrame(
            columns=["DateOnly", "Nickname", "Messages", "Cumulative Messages"]
        )

    daily = (
        df.groupby(["DateOnly", "Nickname"])
        .size()
        .reset_index(name="Messages")
        .sort_values(["Nickname", "DateOnly"])
    )

    daily["Cumulative Messages"] = daily.groupby("Nickname")["Messages"].cumsum()

    return daily


def word_frequency_over_time(
    df: pd.DataFrame,
    words: list[str],
    text_column: str = "Text",
) -> pd.DataFrame:
    if df.empty or text_column not in df.columns or "DateOnly" not in df.columns:
        return pd.DataFrame(columns=["DateOnly", "Word", "Count"])

    clean_words = [str(w).lower().strip() for w in words if str(w).strip()]

    if not clean_words:
        return pd.DataFrame(columns=["DateOnly", "Word", "Count"])

    rows = []

    for word in clean_words:
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)

        temp = df.copy()
        temp["Count"] = (
            temp[text_column]
            .fillna("")
            .astype(str)
            .apply(lambda text: len(pattern.findall(text)))
        )

        daily = temp.groupby("DateOnly")["Count"].sum().reset_index()

        daily["Word"] = word
        rows.append(daily)

    return pd.concat(rows, ignore_index=True)
