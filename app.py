import json
from pathlib import Path
import pandas as pd
import streamlit as st

from cache import (
    delete_cached_chat_set,
    list_cached_chat_sets,
    load_cached_txt_cache,
    load_chats_from_bytes,
    save_uploaded_txt_cache,
    source_mapping_items,
)
from plots import graph_selected, show_wordcloud, small_fig
from ww_functions import (
    DEFAULT_WHATSAPP_PATTERN,
    apply_nickname_mapping,
    count_cooccurrence,
    count_phrase_mentions,
    count_word_by_player,
    create_nickname_mapping,
    top_emojis,
    top_words,
    total_words_by_person,
    word_mentions,
)


CACHE_DIR = Path("cached_chat_uploads")
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_CACHE_NAME = "my_chat_analysis"

# =========================================================
# CONFIG
# =========================================================

st.title("WhatsApp Chat Analyser")


# =========================================================
# SESSION STATE
# =========================================================

if "raw_df" not in st.session_state:
    st.session_state.raw_df = pd.DataFrame()

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if "nickname_map" not in st.session_state:
    st.session_state.nickname_map = pd.DataFrame(
        columns=["Ignore", "WA_Name", "Nickname"]
    )


# =========================================================
# APP HELPERS
# =========================================================


@st.cache_data(show_spinner="Processing chats...")
def cached_load_chats_from_bytes(
    file_data: tuple[tuple[str, bytes], ...],
    source_mapping_items_: tuple[tuple[str, str], ...],
    pattern: str,
) -> pd.DataFrame:
    """
    Streamlit-cached wrapper.

    Permanent cache is handled by cache.py and stores original .txt uploads.
    This cache only speeds up Streamlit reruns.
    """
    return load_chats_from_bytes(
        file_data=file_data,
        source_mapping_items=source_mapping_items_,
        pattern=pattern,
    )


def apply_current_mapping(raw_df: pd.DataFrame):
    """Apply nickname mapping and ignore settings to raw_df."""
    if raw_df.empty:
        st.session_state.df = pd.DataFrame()
        return

    st.session_state.nickname_map = create_nickname_mapping(
        raw_df,
        st.session_state.nickname_map,
    )

    st.session_state.df = apply_nickname_mapping(
        raw_df,
        st.session_state.nickname_map,
        remove_ignored=True,
    )


def get_filtered_df(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Dropdown chat filter. Always returns a dataframe, never a tuple."""
    if df.empty or "Source" not in df.columns:
        return df.copy()

    chat_options = ["All"] + sorted(df["Source"].dropna().astype(str).unique())

    selected_chat = st.selectbox(
        "Filter by chat",
        chat_options,
        key=f"{key_prefix}_chat_filter",
    )

    if selected_chat == "All":
        return df.copy()

    return df[df["Source"].astype(str) == selected_chat].copy()


def choose_word_text_column(df: pd.DataFrame, remove_stopwords: bool) -> str:
    """
    Choose the text column for word-based graphs/stats only.

    Non-word graphs should continue using Text, Sentiment, DateOnly, Hour, etc.
    """
    if remove_stopwords and "FilteredText" in df.columns:
        return "FilteredText"

    return "Text"


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1. Upload chats",
        "2. Name mapping",
        "3. General graphs",
        "4. Person graphs",
        "5. Specific stats",
    ]
)


# =========================================================
# TAB 1 — UPLOAD / CACHE
# =========================================================

with tab1:
    st.header("Upload, merge, and cache chats")

    st.subheader("Upload new chats")

    uploaded_files = st.file_uploader(
        "Upload one or more .txt WhatsApp exports",
        type=["txt"],
        accept_multiple_files=True,
        key="upload_chat_files",
    )

    source_mapping = {}

    if uploaded_files:
        st.write("Uploaded files:")

        for i, uploaded_file in enumerate(uploaded_files):
            source_mapping[uploaded_file.name] = st.text_input(
                f"Chat name for {uploaded_file.name}",
                value=uploaded_file.name.replace(".txt", ""),
                key=f"chat_name_{i}_{uploaded_file.name}",
            )

        if st.button("Process chats", type="primary", key="process_chats_button"):
            pattern = st.session_state.get(
                "upload_regex_pattern",
                DEFAULT_WHATSAPP_PATTERN,
            )
            cache_name = st.session_state.get(
                "cache_name_input",
                "my_chat_analysis",
            )

            saved_cache_name = save_uploaded_txt_cache(
                cache_name=cache_name,
                uploaded_files=uploaded_files,
                source_mapping=source_mapping,
                pattern=pattern,
            )

            file_data = tuple(
                (uploaded_file.name, uploaded_file.getvalue())
                for uploaded_file in uploaded_files
            )

            raw = cached_load_chats_from_bytes(
                file_data=file_data,
                source_mapping_items_=source_mapping_items(source_mapping),
                pattern=pattern,
            )

            st.session_state.raw_df = raw
            apply_current_mapping(raw)

            st.success(
                f"Merged {len(uploaded_files)} file(s), "
                f"txt files as '{saved_cache_name}'."
            )

    st.divider()

    with st.expander("Load or delete cached chats", expanded=False):
        cached_chat_sets = list_cached_chat_sets()

        if cached_chat_sets:
            selected_cache = st.selectbox(
                "Cached chats",
                cached_chat_sets,
                key="cached_chat_dropdown",
            )

            col_load, col_delete = st.columns([1, 1])

            with col_load:
                if st.button("Load cached chat set", key="load_cached_chat_button"):
                    file_data, cached_source_mapping, cached_pattern = (
                        load_cached_txt_cache(selected_cache)
                    )

                    raw = cached_load_chats_from_bytes(
                        file_data=file_data,
                        source_mapping_items_=source_mapping_items(
                            cached_source_mapping
                        ),
                        pattern=cached_pattern,
                    )

                    st.session_state.raw_df = raw
                    apply_current_mapping(raw)

                    st.success(
                        f"Loaded cached chat set '{selected_cache}' with "
                        f"{len(st.session_state.df):,} usable messages."
                    )

            with col_delete:
                if st.button("Delete cached chat set", key="delete_cached_chat_button"):
                    deleted = delete_cached_chat_set(selected_cache)

                    if deleted:
                        st.success(f"Deleted cached chat set '{selected_cache}'.")
                        st.rerun()
                    else:
                        st.error(
                            "Could not delete this cached chat because Windows/OneDrive "
                            "is locking it. Close any File Explorer windows using the "
                            "folder, then try again."
                        )
        else:
            st.info("No cached chat sets yet.")

    st.divider()

    if not st.session_state.df.empty:
        df = st.session_state.df

        st.subheader("Current data")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Messages", f"{len(df):,}")
        c2.metric("Raw messages", f"{len(st.session_state.raw_df):,}")
        c3.metric("People", df["Nickname"].nunique())
        c4.metric("Chats", df["Source"].nunique())

        st.dataframe(df.head(50), width="stretch")

    st.divider()

    with st.expander("Advanced settings", expanded=False):
        st.text_input(
            "WhatsApp message regex pattern",
            value=st.session_state.get(
                "upload_regex_pattern",
                DEFAULT_WHATSAPP_PATTERN,
            ),
            help="Only change this if your WhatsApp export format is different.",
            key="upload_regex_pattern",
        )

        st.text_input(
            "Save uploaded chat set as",
            value=st.session_state.get("cache_name_input", "my_chat_analysis"),
            key="cache_name_input",
        )


# =========================================================
# TAB 2 — NAME MAPPING
# =========================================================

with tab2:
    st.header("Name mapping")

    raw_df = st.session_state.raw_df

    if raw_df.empty:
        st.info("Upload or load chats first.")
    else:
        st.write("Edit **Nickname** and tick **Ignore** for senders you want removed ")

        mapping_df = create_nickname_mapping(
            st.session_state.raw_df,
            st.session_state.nickname_map,
        )

        edited_mapping = st.data_editor(
            mapping_df,
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Ignore": st.column_config.CheckboxColumn(
                    "Ignore",
                    help="Tick to remove this WhatsApp name from graphs and stats.",
                    default=False,
                ),
                "WA_Name": st.column_config.TextColumn(
                    "WhatsApp name",
                    disabled=True,
                ),
                "Nickname": st.column_config.TextColumn(
                    "Nickname",
                    required=True,
                ),
            },
            key="nickname_mapping_editor",
        )

        col_apply, col_download = st.columns([1, 1])

        with col_apply:
            if st.button("Apply mapping", type="primary", key="apply_mapping_button"):
                st.session_state.nickname_map = edited_mapping
                st.session_state.df = apply_nickname_mapping(
                    st.session_state.raw_df,
                    edited_mapping,
                    remove_ignored=True,
                )
                st.success("Mapping applied.")

        with col_download:
            st.download_button(
                "Download mapping JSON",
                data=edited_mapping.to_json(orient="records", indent=2),
                file_name="nickname_mapping.json",
                mime="application/json",
                key="download_mapping_button",
            )

        uploaded_mapping = st.file_uploader(
            "Load saved nickname mapping JSON",
            type=["json"],
            key="upload_mapping_json",
        )

        if uploaded_mapping is not None:
            try:
                loaded_mapping = pd.DataFrame(json.load(uploaded_mapping))
                st.session_state.nickname_map = create_nickname_mapping(
                    st.session_state.raw_df,
                    loaded_mapping,
                )
                st.session_state.df = apply_nickname_mapping(
                    st.session_state.raw_df,
                    st.session_state.nickname_map,
                    remove_ignored=True,
                )
                st.success("Loaded and applied mapping.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load mapping: {exc}")

        st.subheader("Current mapped preview")
        st.dataframe(
            st.session_state.df[["WA_Name", "Nickname", "Source"]]
            .drop_duplicates()
            .sort_values(["Source", "WA_Name"]),
            width="stretch",
            hide_index=True,
        )


# =========================================================
# TAB 3 — GENERAL GRAPHS
# =========================================================

with tab3:
    st.header("General graphs")

    df = st.session_state.df

    if df.empty:
        st.info("Upload or load chats first.")
    else:
        filtered_df = get_filtered_df(df, key_prefix="general")

        remove_stopwords = st.toggle(
            "Remove stopwords for word-based graphs",
            value=True,
            key="general_remove_stopwords",
        )

        word_text_column = choose_word_text_column(
            filtered_df,
            remove_stopwords=remove_stopwords,
        )

        graph_options = [
            "Messages by person",
            "Messages by hour",
            "Messages over time",
            "Average sentiment by person",
            "Total words by person",
            "Top words",
            "Top emojis",
            "Word cloud",
            "View all",
        ]

        selected_graphs = st.multiselect(
            "Choose graph(s) to view",
            graph_options,
            default=["Messages by person"],
            key="general_graph_selector",
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Messages", f"{len(filtered_df):,}")
        c2.metric("People", filtered_df["Nickname"].nunique())
        c3.metric("Chats", filtered_df["Source"].nunique())
        c4.metric("Days", filtered_df["DateOnly"].nunique())

        left, right = st.columns(2)

        if graph_selected(selected_graphs, "Messages by person"):
            with left:
                st.subheader("Messages by person")
                fig, ax = small_fig()
                (
                    filtered_df["Nickname"]
                    .value_counts()
                    .head(15)
                    .sort_values()
                    .plot.barh(ax=ax)
                )
                ax.set_xlabel("Messages")
                ax.set_ylabel("")
                st.pyplot(fig, width="content")

        if graph_selected(selected_graphs, "Messages by hour"):
            with right:
                st.subheader("Messages by hour")
                fig, ax = small_fig()
                (
                    filtered_df.groupby("Hour")
                    .size()
                    .reindex(range(24), fill_value=0)
                    .plot(ax=ax)
                )
                ax.set_xlabel("Hour")
                ax.set_ylabel("Messages")
                st.pyplot(fig, width="content")

        if graph_selected(selected_graphs, "Messages over time"):
            st.subheader("Messages over time")
            fig, ax = small_fig(7, 3)
            filtered_df.groupby("DateOnly").size().plot(ax=ax)
            ax.set_xlabel("Date")
            ax.set_ylabel("Messages")
            st.pyplot(fig, width="content")

        if graph_selected(selected_graphs, "Average sentiment by person"):
            st.subheader("Average sentiment by person")
            fig, ax = small_fig(7, 3)
            (
                filtered_df.groupby("Nickname")["Sentiment"]
                .mean()
                .sort_values()
                .tail(20)
                .plot.barh(ax=ax)
            )
            ax.set_xlabel("Average sentiment")
            ax.set_ylabel("")
            st.pyplot(fig, width="content")

        if graph_selected(selected_graphs, "Total words by person"):
            st.subheader("Total words by person")
            st.caption(
                "Uses original Text, not FilteredText, because all words matter here."
            )
            total_word_df = total_words_by_person(filtered_df, text_column="Text")
            st.dataframe(total_word_df, width="stretch", hide_index=True)

        if graph_selected(selected_graphs, "Top words"):
            st.subheader("Top words")
            st.caption(f"Using text column: {word_text_column}")
            st.dataframe(
                top_words(filtered_df, n=30, text_column=word_text_column),
                width="stretch",
                hide_index=True,
            )

        if graph_selected(selected_graphs, "Top emojis"):
            st.subheader("Top emojis")
            st.dataframe(
                top_emojis(filtered_df, n=30, text_column="Text"),
                width="stretch",
                hide_index=True,
            )

        if graph_selected(selected_graphs, "Word cloud"):
            st.subheader("Word cloud")
            st.caption(f"Using text column: {word_text_column}")
            show_wordcloud(
                " ".join(filtered_df[word_text_column].fillna("").astype(str)),
                "Overall word cloud",
                st=st,
            )


# =========================================================
# TAB 4 — PERSON GRAPHS
# =========================================================

with tab4:
    st.header("Per Person graphs")

    df = st.session_state.df

    if df.empty:
        st.info("Upload or load chats first.")
    else:
        filtered_df = get_filtered_df(df, key_prefix="person_graphs")

        remove_stopwords = st.toggle(
            "Remove stopwords for word-based graphs",
            value=True,
            key="person_remove_stopwords",
        )

        word_text_column = choose_word_text_column(
            filtered_df,
            remove_stopwords=remove_stopwords,
        )

        people = sorted(filtered_df["Nickname"].dropna().astype(str).unique())

        if not people:
            st.info("No people available after filtering.")
        else:
            selected_person = st.selectbox(
                "Choose a person",
                people,
                key="person_graph_person_selector",
            )

            person_df = filtered_df[
                filtered_df["Nickname"].astype(str) == selected_person
            ].copy()

            person_graph_options = [
                "Messages by hour",
                "Messages over time",
                "Sentiment over time",
                "Total words",
                "Top words",
                "Top emojis",
                "Word cloud",
                "View all",
            ]

            selected_person_graphs = st.multiselect(
                "Choose graph(s) to view",
                person_graph_options,
                default=["Messages by hour"],
                key="person_graph_selector",
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Messages", f"{len(person_df):,}")
            c2.metric(
                "Average sentiment",
                round(person_df["Sentiment"].mean(), 3) if len(person_df) else 0,
            )
            c3.metric("Active days", person_df["DateOnly"].nunique())

            left, right = st.columns(2)

            if graph_selected(selected_person_graphs, "Messages by hour"):
                with left:
                    st.subheader("Messages by hour")
                    fig, ax = small_fig()
                    (
                        person_df.groupby("Hour")
                        .size()
                        .reindex(range(24), fill_value=0)
                        .plot(ax=ax)
                    )
                    ax.set_xlabel("Hour")
                    ax.set_ylabel("Messages")
                    st.pyplot(fig, width="content")

            if graph_selected(selected_person_graphs, "Messages over time"):
                with right:
                    st.subheader("Messages over time")
                    fig, ax = small_fig()
                    person_df.groupby("DateOnly").size().plot(ax=ax)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Messages")
                    st.pyplot(fig, width="content")

            if graph_selected(selected_person_graphs, "Sentiment over time"):
                st.subheader("Sentiment over time")
                fig, ax = small_fig(7, 3)
                person_df.groupby("DateOnly")["Sentiment"].mean().plot(ax=ax)
                ax.set_xlabel("Date")
                ax.set_ylabel("Average sentiment")
                st.pyplot(fig, width="content")

            if graph_selected(selected_person_graphs, "Total words"):
                st.subheader("Total words")
                st.caption(
                    "Uses original Text, not FilteredText, because all words matter here."
                )
                total_word_count = int(
                    person_df["Text"]
                    .fillna("")
                    .astype(str)
                    .str.split()
                    .apply(len)
                    .sum()
                )
                st.metric("Total words", f"{total_word_count:,}")

            if graph_selected(selected_person_graphs, "Top words"):
                st.subheader("Top words")
                st.caption(f"Using text column: {word_text_column}")
                st.dataframe(
                    top_words(person_df, n=30, text_column=word_text_column),
                    width="stretch",
                    hide_index=True,
                )

            if graph_selected(selected_person_graphs, "Top emojis"):
                st.subheader("Top emojis")
                st.dataframe(
                    top_emojis(person_df, n=30, text_column="Text"),
                    width="stretch",
                    hide_index=True,
                )

            if graph_selected(selected_person_graphs, "Word cloud"):
                st.subheader("Word cloud")
                st.caption(f"Using text column: {word_text_column}")
                show_wordcloud(
                    " ".join(person_df[word_text_column].fillna("").astype(str)),
                    f"Word cloud for {selected_person}",
                    st=st,
                )


# =========================================================
# TAB 5 — SPECIFIC STATS
# =========================================================

with tab5:
    st.header("Specific stats")

    df = st.session_state.df

    if df.empty:
        st.info("Upload or load chats first.")
    else:
        filtered_df = get_filtered_df(df, key_prefix="specific_stats")

        remove_stopwords = st.toggle(
            "Remove stopwords for word-based stats",
            value=False,
            key="specific_stats_remove_stopwords",
            help=(
                "Default is off here because exact word/phrase searches usually need "
                "the original message text."
            ),
        )

        stat_text_column = choose_word_text_column(
            filtered_df,
            remove_stopwords=remove_stopwords,
        )

        st.caption(f"Stats are currently using text column: {stat_text_column}")

        with st.expander(
            "How many messages contain two words/phrases together?",
            expanded=True,
        ):
            col1, col2 = st.columns(2)

            word1 = col1.text_input(
                "First word or phrase",
                key="cooccurrence_word_1",
            )

            word2 = col2.text_input(
                "Second word or phrase",
                key="cooccurrence_word_2",
            )

            if st.button(
                "Count messages containing both",
                key="count_cooccurrence_button",
            ):
                count = count_cooccurrence(
                    filtered_df,
                    word1,
                    word2,
                    text_column=stat_text_column,
                )
                st.metric("Messages containing both", count)

        with st.expander("How many times was a word used, and by who?"):
            target_word = st.text_input(
                "Word to analyse",
                key="word_mentions_input",
            )

            if st.button("Analyse word", key="analyse_word_button"):
                result = word_mentions(
                    filtered_df,
                    target_word,
                    text_column=stat_text_column,
                )

                st.metric("Total mentions", result["total_mentions"])

                st.dataframe(
                    pd.DataFrame(
                        result["mentions_by_player"].items(),
                        columns=["Person", "Mentions"],
                    ),
                    width="stretch",
                    hide_index=True,
                )

        with st.expander("How many times did one person use a word?"):
            people = sorted(filtered_df["Nickname"].dropna().astype(str).unique())

            if people:
                selected_person = st.selectbox(
                    "Person",
                    people,
                    key="specific_person_selector",
                )

                selected_word = st.text_input(
                    "Word",
                    key="specific_person_word_input",
                )

                if st.button(
                    "Count person's word usage",
                    key="count_person_word_button",
                ):
                    count = count_word_by_player(
                        filtered_df,
                        selected_person,
                        selected_word,
                        text_column=stat_text_column,
                    )

                    st.metric("Mentions", count)
            else:
                st.info("No people available after filtering.")

        with st.expander("How many times was an exact phrase used?"):
            phrase = st.text_input(
                "Exact phrase",
                key="exact_phrase_input",
            )

            if st.button("Count phrase", key="count_phrase_button"):
                count = count_phrase_mentions(
                    filtered_df,
                    phrase,
                    text_column=stat_text_column,
                )
                st.metric("Phrase mentions", count)
