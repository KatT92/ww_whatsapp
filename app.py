import json
from pathlib import Path
import pandas as pd
import streamlit as st

from cache import (
    delete_cached_chat_set,
    list_cached_chat_sets,
    load_cached_txt_cache,
    load_chats_from_bytes as load_chats_from_bytes_uncached,
    save_uploaded_txt_cache,
    source_mapping_items,
)

from parse_data import (
    DEFAULT_WHATSAPP_PATTERN,
    BRACKETED_WHATSAPP_PATTERN,
    WHATSAPP_PATTERNS,
    apply_nickname_mapping,
    create_nickname_mapping,
)

from stats import (
    count_cooccurrence,
    count_phrase_mentions,
    count_word_by_player,
    top_emojis,
    top_words,
    total_words_by_person,
    word_mentions,
    general_summary_stats,
    longest_messages,
    ngrams,
    most_active_hour_all_players,
    most_active_day_all_players,
    messages_between_players,
    messages_between_players_by_day,
    sentiment_from_player_to_player_by_day,
    sentiment_about_player_from_all,
    sentiment_from_player_to_all,
    plot_normalised_mention_heatmap,
    plot_most_active_hour_by_player,
    plot_most_active_day_by_player,
    cumulative_messages_over_time,
    word_frequency_over_time,
)

from plots import show_wordcloud, small_fig, graph_selected


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
    return load_chats_from_bytes_uncached(
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
        "4. Per Person graphs",
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
                            "Could not delete this cached chat, please close the file and try again"
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
        pattern_choice = st.selectbox(
            "WhatsApp export format",
            list(WHATSAPP_PATTERNS.keys()),
            index=0,
            key="pattern_choice",
        )

        if pattern_choice == "Custom pattern":
            pattern = st.text_input(
                "Custom WhatsApp regex pattern",
                value=st.session_state.get(
                    "custom_upload_regex_pattern",
                    DEFAULT_WHATSAPP_PATTERN,
                ),
                help="Must return four groups: Date, Time, WA_Name, Text.",
                key="custom_upload_regex_pattern",
            )
        else:
            pattern = WHATSAPP_PATTERNS[pattern_choice]

            st.code(pattern, language="python")

        st.session_state["upload_regex_pattern"] = pattern

        st.text_input(
            "Save uploaded chat set as",
            value=st.session_state.get("cache_name_input", "my_chat_analysis"),
            key="cache_name_input",
        )

    st.divider()

    col1, col2 = st.columns([8,1])

    with col2:
        with st.popover("ℹ️ Help"):
            st.markdown("""
    ### HELP

    - So, you're here because you need help.

    - If you're having issues with the user experience, that's a you problem, I'm a data scientist not a front-end dev.
    - If the app breaks it might be a you problem, consider not doing what you just did, but if you think it's a me problem, maybe let me know depending on how broken it is.
    - If you want another graph, I'll get to it in 4-5 business weeks.                  
    - If the file isn't uploaded properly try the 2 different patterns in advanced settings
    - If that doesnt work, try copy and pasting a few lines of the .txt file into chatgpt/a regex formatter and asking what the regex pattern should be and putting it in the custom slot.
    - If in doubt, press the big red buttons.
                        
    """)

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
        # Chat dropdown filter: this filters rows/messages only.
        # It is not the stopword filter.
        chat_df = get_filtered_df(df, key_prefix="general")

        # Stopword toggle: this only chooses the text column for word-based outputs.
        remove_stopwords = st.toggle(
            "Remove stopwords for word-based graphs",
            value=True,
            key="general_remove_stopwords",
        )

        word_text_column = choose_word_text_column(
            chat_df,
            remove_stopwords=remove_stopwords,
        )

        graph_options = [
            "Summary stats",
            "Messages by person",
            "Messages by hour",
            "Messages over time",
            "Average sentiment by person",
            "Total words by person",
            "Longest messages",
            "Top words",
            "Top bigrams",
            "Top trigrams",
            "Top emojis",
            "Word cloud",
            "Most active hour by player",
            "Most active day by player",
            "View all",
        ]

        selected_graphs = st.multiselect(
            "Choose graph(s) to view",
            graph_options,
            default=["Messages by person"],
            key="general_graph_selector",
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Messages", f"{len(chat_df):,}")
        c2.metric("People", chat_df["Nickname"].nunique())
        c3.metric("Chats", chat_df["Source"].nunique())
        c4.metric("Days", chat_df["DateOnly"].nunique())

        if graph_selected(selected_graphs, "Summary stats"):
            st.subheader("Summary stats")

            st.dataframe(
                general_summary_stats(chat_df),
                width="stretch",
                hide_index=True,
            )

        left, right = st.columns(2)

        if graph_selected(selected_graphs, "Messages by person"):
            with left:
                st.subheader("Messages by person")
                fig, ax = small_fig()
                (
                    chat_df["Nickname"]
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
                    chat_df.groupby("Hour")
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
            chat_df.groupby("DateOnly").size().plot(ax=ax)
            ax.set_xlabel("Date")
            ax.set_ylabel("Messages")
            st.pyplot(fig, width="content")

        if graph_selected(selected_graphs, "Word cloud"):
            st.subheader("Word cloud")
            st.caption(f"Using text column: {word_text_column}")
            show_wordcloud(
                " ".join(chat_df[word_text_column].fillna("").astype(str)),
                "Overall word cloud",
                st=st,
            )

        if graph_selected(selected_graphs, "Average sentiment by person"):
            st.subheader("Average sentiment by person")
            fig, ax = small_fig(7, 3)
            (
                chat_df.groupby("Nickname")["Sentiment"]
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

            total_word_df = total_words_by_person(chat_df, text_column="Text")
            st.dataframe(total_word_df, width="stretch", hide_index=True)

        if graph_selected(selected_graphs, "Longest messages"):
            st.subheader("Longest messages")

            st.dataframe(
                longest_messages(chat_df, n=10, text_column="Text"),
                width="stretch",
                hide_index=True,
            )

        if graph_selected(selected_graphs, "Top words"):
            st.subheader("Top words")
            st.caption(f"Using text column: {word_text_column}")
            st.dataframe(
                top_words(chat_df, n=30, text_column=word_text_column),
                width="stretch",
                hide_index=True,
            )

        if graph_selected(selected_graphs, "Top bigrams"):
            st.subheader("Top bigrams")
            st.caption(f"Using text column: {word_text_column}")
            st.dataframe(
                ngrams(chat_df, ngram_size=2, top_n=30, text_column=word_text_column),
                width="stretch",
                hide_index=True,
            )

        if graph_selected(selected_graphs, "Top trigrams"):
            st.subheader("Top trigrams")
            st.caption(f"Using text column: {word_text_column}")
            st.dataframe(
                ngrams(chat_df, ngram_size=3, top_n=30, text_column=word_text_column),
                width="stretch",
                hide_index=True,
            )

        if graph_selected(selected_graphs, "Top emojis"):
            st.subheader("Top emojis")

            st.dataframe(
                top_emojis(chat_df, n=30, text_column="Text"),
                width="stretch",
                hide_index=True,
            )

        if graph_selected(selected_graphs, "Most active hour by player"):
            st.subheader("Most active hour by player")

            active_hour_df = most_active_hour_all_players(chat_df)

            if not active_hour_df.empty:
                fig, ax = small_fig(7, 3)

                active_hour_df.sort_values("Messages").plot.barh(
                    x="Nickname",
                    y="Messages",
                    ax=ax,
                    legend=False,
                )

                ax.set_xlabel("Messages at most active hour")
                ax.set_ylabel("Player")
                ax.set_title("Most active hour by player")

                st.pyplot(fig, width="content")

        if graph_selected(selected_graphs, "Most active day by player"):
            st.subheader("Most active day by player")

            active_day_df = most_active_day_all_players(chat_df)

            if not active_day_df.empty:
                fig, ax = small_fig(7, 3)

                active_day_df.sort_values("Messages").plot.barh(
                    x="Nickname",
                    y="Messages",
                    ax=ax,
                    legend=False,
                )

                ax.set_xlabel("Messages on most active day")
                ax.set_ylabel("Player")
                ax.set_title("Most active day by player")

                st.pyplot(fig, width="content")

                fig, ax, active_day_df = plot_most_active_day_by_player(
                    chat_df,
                    min_messages=5,
                )

                st.pyplot(fig, width="content")

        if graph_selected(selected_graphs, "Most active hour by player"):
            st.subheader("Most active hour by player")

            fig, ax, active_df = plot_most_active_hour_by_player(
                chat_df,
                min_messages=5,
            )

            st.pyplot(fig, width="content")

            st.dataframe(
                active_df,
                width="stretch",
                hide_index=True,
            )

        if graph_selected(selected_graphs, "Normalised mention heatmap"):
            st.subheader("Special graph for Louis")

            fig, ax, matrix_df = plot_normalised_mention_heatmap(
                chat_df,
                st.session_state.nickname_map,
            )

            st.pyplot(fig, width="content")


# =========================================================
# TAB 4 — PERSON GRAPHS
# =========================================================

with tab4:
    st.header("Per Person graphs")

    df = st.session_state.df

    if df.empty:
        st.info("Upload or load chats first.")
    else:
        chat_df = get_filtered_df(df, key_prefix="person_graphs")

        remove_stopwords = st.toggle(
            "Remove stopwords for word-based graphs",
            value=True,
            key="person_remove_stopwords",
        )

        word_text_column = choose_word_text_column(
            chat_df,
            remove_stopwords=remove_stopwords,
        )

        people = sorted(chat_df["Nickname"].dropna().astype(str).unique())

        if not people:
            st.info("No people available after filtering.")
        else:
            selected_people = st.multiselect(
                "Choose one or two people",
                people,
                default=people[:1],
                max_selections=2,
                key="person_graph_people_selector",
            )

            if not selected_people:
                st.info("Select at least one person.")
            else:
                person_df = chat_df[
                    chat_df["Nickname"].astype(str).isin(selected_people)
                ].copy()

                person_graph_options = [
                    "Messages by hour",
                    "Messages over time",
                    "Cumulative messages over time",
                    "Sentiment over time",
                    "Total words",
                    "Top words",
                    "Top bigrams",
                    "Top trigrams",
                    "Top emojis",
                    "Word cloud",
                    "Messages between players",
                    "Messages between selected players by day",
                    "Sentiment from one to the other by day",
                    "Sentiment about selected player from all",
                    "Sentiment from selected player to others",
                    "View all",
                ]
                selected_person_graphs = st.multiselect(
                    "Choose graph(s) to view",
                    person_graph_options,
                    default=["Messages by hour"],
                    key="person_graph_selector",
                )

                summary = (
                    person_df.groupby("Nickname")
                    .agg(
                        Messages=("Text", "count"),
                        Avg_Sentiment=("Sentiment", "mean"),
                        Active_Days=("DateOnly", "nunique"),
                    )
                    .reset_index()
                )
                summary["Avg_Sentiment"] = summary["Avg_Sentiment"].round(3)

                st.subheader("Selected player summary")
                st.dataframe(summary, width="stretch", hide_index=True)

                if graph_selected(selected_person_graphs, "Messages by hour"):
                    st.subheader("Messages by hour")

                    hourly = (
                        person_df.groupby(["Hour", "Nickname"])
                        .size()
                        .reset_index(name="Messages")
                    )

                    pivot = (
                        hourly.pivot(
                            index="Hour",
                            columns="Nickname",
                            values="Messages",
                        )
                        .reindex(range(24), fill_value=0)
                        .fillna(0)
                    )

                    fig, ax = small_fig(7, 3)
                    pivot.plot(ax=ax)
                    ax.set_xlabel("Hour")
                    ax.set_ylabel("Messages")
                    st.pyplot(fig, width="content")

                if graph_selected(selected_person_graphs, "Messages over time"):
                    st.subheader("Messages over time")

                    daily = (
                        person_df.groupby(["DateOnly", "Nickname"])
                        .size()
                        .reset_index(name="Messages")
                    )

                    pivot = daily.pivot(
                        index="DateOnly",
                        columns="Nickname",
                        values="Messages",
                    ).fillna(0)

                    fig, ax = small_fig(7, 3)
                    pivot.plot(ax=ax)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Messages")
                    st.pyplot(fig, width="content")

                if graph_selected(
                    selected_person_graphs, "Cumulative messages over time"
                ):
                    st.subheader("Cumulative messages over time")

                    cumulative_df = cumulative_messages_over_time(person_df)

                    if cumulative_df.empty:
                        st.info("No cumulative message data available.")
                    else:
                        pivot = (
                            cumulative_df.pivot(
                                index="DateOnly",
                                columns="Nickname",
                                values="Cumulative Messages",
                            )
                            .fillna(method="ffill")
                            .fillna(0)
                        )

                        fig, ax = small_fig(7, 3)

                        pivot.plot(ax=ax)

                        ax.set_xlabel("Date")
                        ax.set_ylabel("Cumulative messages")

                        st.pyplot(fig, width="content")

                        st.dataframe(
                            cumulative_df,
                            width="stretch",
                            hide_index=True,
                        )

                if graph_selected(selected_person_graphs, "Sentiment over time"):
                    st.subheader("Sentiment over time")

                    sentiment = (
                        person_df.groupby(["DateOnly", "Nickname"])["Sentiment"]
                        .mean()
                        .reset_index()
                    )

                    pivot = sentiment.pivot(
                        index="DateOnly",
                        columns="Nickname",
                        values="Sentiment",
                    )

                    fig, ax = small_fig(7, 3)
                    pivot.plot(ax=ax)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Average sentiment")
                    st.pyplot(fig, width="content")

                if graph_selected(selected_person_graphs, "Total words"):
                    st.subheader("Total words")
      

                    total_words = (
                        person_df.assign(
                            WordCount=person_df["Text"]
                            .fillna("")
                            .astype(str)
                            .str.split()
                            .apply(len)
                        )
                        .groupby("Nickname")["WordCount"]
                        .sum()
                        .sort_values(ascending=False)
                        .reset_index()
                    )

                    total_words.columns = ["Nickname", "Total words"]
                    st.dataframe(total_words, width="stretch", hide_index=True)

                if graph_selected(selected_person_graphs, "Top words"):
                    st.subheader("Top words")
                    st.caption(f"Using text column: {word_text_column}")

                    for person in selected_people:
                        st.markdown(f"**{person}**")
                        this_person_df = person_df[
                            person_df["Nickname"].astype(str) == person
                        ]

                        st.dataframe(
                            top_words(
                                this_person_df,
                                n=30,
                                text_column=word_text_column,
                            ),
                            width="stretch",
                            hide_index=True,
                        )

                if graph_selected(selected_person_graphs, "Top emojis"):
                    st.subheader("Top emojis")


                    for person in selected_people:
                        st.markdown(f"**{person}**")
                        this_person_df = person_df[
                            person_df["Nickname"].astype(str) == person
                        ]

                        st.dataframe(
                            top_emojis(
                                this_person_df,
                                n=30,
                                text_column="Text",
                            ),
                            width="stretch",
                            hide_index=True,
                        )

                if graph_selected(selected_person_graphs, "Word cloud"):
                    st.subheader("Word cloud")
                    st.caption(f"Using text column: {word_text_column}")

                    for person in selected_people:
                        this_person_df = person_df[
                            person_df["Nickname"].astype(str) == person
                        ]

                        show_wordcloud(
                            " ".join(
                                this_person_df[word_text_column].fillna("").astype(str)
                            ),
                            f"Word cloud for {person}",
                            st=st,
                        )

                if graph_selected(selected_person_graphs, "Top bigrams"):
                    st.subheader("Top bigrams")
                    st.caption(f"Using text column: {word_text_column}")

                    for person in selected_people:
                        st.markdown(f"**{person}**")

                        this_person_df = person_df[
                            person_df["Nickname"].astype(str) == person
                        ]

                        st.dataframe(
                            ngrams(
                                this_person_df,
                                ngram_size=2,
                                top_n=30,
                                text_column=word_text_column,
                            ),
                            width="stretch",
                            hide_index=True,
                        )

                if graph_selected(selected_person_graphs, "Top trigrams"):
                    st.subheader("Top trigrams")
                    st.caption(f"Using text column: {word_text_column}")

                    for person in selected_people:
                        st.markdown(f"**{person}**")

                        this_person_df = person_df[
                            person_df["Nickname"].astype(str) == person
                        ]

                        st.dataframe(
                            ngrams(
                                this_person_df,
                                ngram_size=3,
                                top_n=30,
                                text_column=word_text_column,
                            ),
                            width="stretch",
                            hide_index=True,
                        )

                if len(selected_people) == 2 and graph_selected(
                    selected_person_graphs, "Messages between players"
                ):
                    st.subheader("Messages between players")

                    player_a, player_b = selected_people

                    interaction_df = messages_between_players(
                        chat_df,
                        player_a,
                        player_b,
                        text_column="Text",
                    )

                    if interaction_df.empty:
                        st.info("No messages found between these players.")
                    else:
                        st.dataframe(
                            interaction_df,
                            width="stretch",
                            hide_index=True,
                        )

                        fig, ax = small_fig(6, 3)

                        interaction_df.plot.bar(
                            x="Direction",
                            y="Messages",
                            ax=ax,
                            legend=False,
                        )

                        ax.set_xlabel("Direction")
                        ax.set_ylabel("Messages")
                        ax.set_title("Messages between players")

                        st.pyplot(fig, width="content")

                if len(selected_people) == 2 and graph_selected(
                    selected_person_graphs,
                    "Messages between selected players by day",
                ):
                    st.subheader("Messages between selected players by day")

                    player_a, player_b = selected_people

                    messages = messages_between_players_by_day(
                        chat_df,
                        player_a,
                        player_b,
                        st.session_state.nickname_map,
                        text_column="Text",
                    )

                    if messages.empty:
                        st.info("No messages found between these players.")
                    else:
                        pivot = messages.pivot(
                            index="DateOnly",
                            columns="Direction",
                            values="Messages",
                        ).fillna(0)

                        fig, ax = small_fig(7, 3)
                        pivot.plot(ax=ax)
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Messages mentioning player")
                        st.pyplot(fig, width="content")

                        st.dataframe(
                            messages,
                            width="stretch",
                            hide_index=True,
                        )

                if len(selected_people) == 2 and graph_selected(
                    selected_person_graphs,
                    "Sentiment from one to the other by day",
                ):
                    st.subheader("Sentiment from one player to the other by day")

                    player_a, player_b = selected_people

                    direction = st.radio(
                        "Direction",
                        [
                            f"{player_a} → {player_b}",
                            f"{player_b} → {player_a}",
                        ],
                        horizontal=True,
                        key="sentiment_direction_selector",
                    )

                    if direction == f"{player_a} → {player_b}":
                        from_player, to_player = player_a, player_b
                    else:
                        from_player, to_player = player_b, player_a

                    daily_sentiment = sentiment_from_player_to_player_by_day(
                        chat_df,
                        from_player,
                        to_player,
                        st.session_state.nickname_map,
                        text_column="Text",
                    )

                    if daily_sentiment.empty:
                        st.info("No messages found in that direction.")
                    else:
                        fig, ax = small_fig(7, 3)

                        daily_sentiment.plot(
                            x="DateOnly",
                            y="Average Sentiment",
                            ax=ax,
                            legend=False,
                        )

                        ax.set_xlabel("Date")
                        ax.set_ylabel("Average sentiment")
                        st.pyplot(fig, width="content")

                        st.dataframe(
                            daily_sentiment,
                            width="stretch",
                            hide_index=True,
                        )

                if graph_selected(
                    selected_person_graphs,
                    "Sentiment about selected player from all",
                ):
                    st.subheader(
                        "Average sentiment about selected player from all players"
                    )

                    target_player = st.selectbox(
                        "Target player",
                        selected_people,
                        key="target_player_sentiment_from_all",
                    )

                    about_df = sentiment_about_player_from_all(
                        chat_df,
                        target_player,
                        st.session_state.nickname_map,
                        text_column="Text",
                    )

                    if about_df.empty:
                        st.info("No messages mentioning this player.")
                    else:
                        fig, ax = small_fig(7, 3)

                        about_df.sort_values("Average Sentiment").plot.barh(
                            x="From",
                            y="Average Sentiment",
                            ax=ax,
                            legend=False,
                        )

                        ax.set_xlabel("Average sentiment")
                        ax.set_ylabel("From")
                        st.pyplot(fig, width="content")

                        st.dataframe(
                            about_df,
                            width="stretch",
                            hide_index=True,
                        )

                if graph_selected(
                    selected_person_graphs,
                    "Sentiment from selected player to others",
                ):
                    st.subheader(
                        "Average sentiment from selected player towards others"
                    )

                    from_player = st.selectbox(
                        "From player",
                        selected_people,
                        key="from_player_sentiment_to_others",
                    )

                    towards_df = sentiment_from_player_to_all(
                        chat_df,
                        from_player,
                        st.session_state.nickname_map,
                        text_column="Text",
                    )

                    if towards_df.empty:
                        st.info("No messages found from this player mentioning others.")
                    else:
                        fig, ax = small_fig(7, 3)

                        towards_df.sort_values("Average Sentiment").plot.barh(
                            x="To",
                            y="Average Sentiment",
                            ax=ax,
                            legend=False,
                        )

                        ax.set_xlabel("Average sentiment")
                        ax.set_ylabel("To")
                        st.pyplot(fig, width="content")

                        st.dataframe(
                            towards_df,
                            width="stretch",
                            hide_index=True,
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
        chat_df = get_filtered_df(df, key_prefix="specific_stats")

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
            chat_df,
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
                    chat_df,
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
                    chat_df,
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
            people = sorted(chat_df["Nickname"].dropna().astype(str).unique())

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
                        chat_df,
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
                    chat_df,
                    phrase,
                    text_column=stat_text_column,
                )
                st.metric("Phrase mentions", count)

        with st.expander("Word frequency over time", expanded=True):
            words_input = st.text_input(
                "Word or words to track",
                help="Enter one or more words separated by commas, e.g. wolf, burn, vote",
                key="word_frequency_over_time_input",
            )

            if st.button(
                "Show word frequency over time", key="word_frequency_over_time_button"
            ):
                words = [
                    word.strip() for word in words_input.split(",") if word.strip()
                ]

                freq_df = word_frequency_over_time(
                    chat_df,
                    words,
                    text_column=stat_text_column,
                )

                if freq_df.empty:
                    st.info("No word frequency data found.")
                else:
                    pivot = freq_df.pivot(
                        index="DateOnly",
                        columns="Word",
                        values="Count",
                    ).fillna(0)

                    fig, ax = small_fig(7, 3)

                    pivot.plot(ax=ax)

                    ax.set_xlabel("Date")
                    ax.set_ylabel("Word count")

                    st.pyplot(fig, width="content")
