import json
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from ww_functions import (
    DEFAULT_WHATSAPP_PATTERN,
    apply_nickname_mapping,
    count_cooccurrence,
    count_phrase_mentions,
    count_word_by_player,
    create_nickname_mapping,
    merge_whatsapp_chats,
    prepare_dataframe,
    top_emojis,
    top_words,
    word_mentions,
)


st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.title("WhatsApp Chat Analyzer")


# =====================================================
# SESSION STATE
# =====================================================

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if "nickname_map" not in st.session_state:
    st.session_state.nickname_map = pd.DataFrame(columns=["WA_Name", "Nickname"])


# =====================================================
# HELPERS
# =====================================================

def small_fig(width=6, height=3):
    return plt.subplots(figsize=(width, height))


def show_wordcloud(text: str, title: str = "Word cloud"):
    if not text or not text.strip():
        st.info("No text available for this word cloud.")
        return

    wc = WordCloud(width=900, height=450, background_color="white").generate(text)
    fig, ax = small_fig(6, 3)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig, use_container_width=False)


def graph_selected(selected_graphs, graph_name):
    return "View all" in selected_graphs or graph_name in selected_graphs


def get_filtered_df(df, key_prefix):
    chat_options = ["All"] + sorted(df["Source"].dropna().astype(str).unique())

    selected_chat = st.selectbox(
        "Filter by chat",
        chat_options,
        key=f"{key_prefix}_chat_filter",
    )

    if selected_chat == "All":
        return df.copy(), selected_chat

    return df[df["Source"].astype(str) == selected_chat].copy(), selected_chat


def choose_filtered_or_all(filtered_df, full_df, selected_chat, key_prefix):
    """
    Adds a toggle to choose whether to use the filtered chat or all chats.
    Defaults to filtered.
    """
    use_filtered = st.toggle(
        "Use chat filter",
        value=True,
        key=f"{key_prefix}_use_chat_filter_toggle",
        help="Turn this off to use all uploaded chats instead of the selected chat filter.",
    )

    if use_filtered:
        active_df = filtered_df.copy()
        st.caption(
            f"Using filtered chat: {selected_chat} ({len(active_df):,} messages)."
        )
    else:
        active_df = full_df.copy()
        st.caption(f"Using all chats ({len(active_df):,} messages).")

    return active_df


# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1. Upload chats",
        "2. Name mapping",
        "3. General graphs",
        "4. Person graphs",
        "5. Specific stats",
    ]
)


# =====================================================
# TAB 1 - UPLOAD CHATS
# =====================================================

with tab1:
    st.header("Upload multiple chats and name them")

    pattern = st.text_input(
        "WhatsApp message regex pattern",
        value=DEFAULT_WHATSAPP_PATTERN,
        help="Change this only if your WhatsApp export format is different.",
        key="upload_regex_pattern",
    )

    uploaded_files = st.file_uploader(
        "Upload .txt WhatsApp exports",
        type=["txt"],
        accept_multiple_files=True,
        key="upload_chat_files",
    )

    source_mapping = {}

    if uploaded_files:
        st.subheader("Name each uploaded chat")

        for i, uploaded_file in enumerate(uploaded_files):
            source_mapping[uploaded_file.name] = st.text_input(
                f"Chat name for {uploaded_file.name}",
                value=uploaded_file.name.replace(".txt", ""),
                key=f"chat_name_{i}_{uploaded_file.name}",
            )

        if st.button("Process chats", type="primary", key="process_chats_button"):
            merged = merge_whatsapp_chats(uploaded_files, source_mapping, pattern)
            merged = prepare_dataframe(merged)

            st.session_state.nickname_map = create_nickname_mapping(
                merged,
                st.session_state.nickname_map,
            )

            merged = apply_nickname_mapping(
                merged,
                st.session_state.nickname_map,
            )

            st.session_state.df = merged

            st.success(
                f"Loaded {len(merged):,} messages from {merged['Source'].nunique()} chat(s)."
            )

    if not st.session_state.df.empty:
        df = st.session_state.df

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Messages", f"{len(df):,}")
        c2.metric("Original names", df["WA_Name"].nunique())
        c3.metric("Mapped names", df["Nickname"].nunique())
        c4.metric("Chats", df["Source"].nunique())

        st.subheader("Preview")
        st.dataframe(df.head(50), use_container_width=True)


# =====================================================
# TAB 2 - NAME MAPPING
# =====================================================

with tab2:
    st.header("Map WhatsApp names to nicknames")

    df = st.session_state.df

    if df.empty:
        st.info("Upload chats first.")
    else:
        st.write(
            "Edit the **Nickname** column. Every graph and stat will use this mapping once you click **Apply mapping**."
        )

        mapping_df = create_nickname_mapping(df, st.session_state.nickname_map)

        edited_mapping = st.data_editor(
            mapping_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
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

        col_a, col_b = st.columns([1, 1])

        with col_a:
            if st.button("Apply mapping", type="primary", key="apply_mapping_button"):
                st.session_state.nickname_map = edited_mapping
                st.session_state.df = apply_nickname_mapping(df, edited_mapping)
                st.success("Mapping applied.")

        with col_b:
            st.download_button(
                "Download mapping JSON",
                data=edited_mapping.to_json(orient="records", indent=2),
                file_name="nickname_mapping.json",
                mime="application/json",
                key="download_mapping_button",
            )

        uploaded_mapping = st.file_uploader(
            "Load a saved nickname mapping JSON",
            type=["json"],
            key="upload_mapping_json",
        )

        if uploaded_mapping is not None:
            try:
                loaded = pd.DataFrame(json.load(uploaded_mapping))
                st.session_state.nickname_map = create_nickname_mapping(df, loaded)
                st.session_state.df = apply_nickname_mapping(
                    df,
                    st.session_state.nickname_map,
                )
                st.success("Loaded and applied mapping.")
            except Exception as exc:
                st.error(f"Could not load mapping: {exc}")

        st.subheader("Current mapped preview")
        st.dataframe(
            st.session_state.df[["WA_Name", "Nickname"]]
            .drop_duplicates()
            .sort_values("WA_Name"),
            use_container_width=True,
            hide_index=True,
        )


# =====================================================
# TAB 3 - GENERAL GRAPHS
# =====================================================

with tab3:
    st.header("General graphs")

    df = st.session_state.df

    if df.empty:
        st.info("Upload chats first.")
    else:
        filtered, selected_chat = get_filtered_df(df, key_prefix="general")
        analysis_df = choose_filtered_or_all(
            filtered,
            df,
            selected_chat,
            key_prefix="general_graphs",
        )

        graph_options = [
            "Messages by person",
            "Messages by hour",
            "Messages over time",
            "Average sentiment by person",
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
        c1.metric("Messages", f"{len(analysis_df):,}")
        c2.metric("People", analysis_df["Nickname"].nunique())
        c3.metric("Chats", analysis_df["Source"].nunique())
        c4.metric("Days", analysis_df["DateOnly"].nunique())

        left, right = st.columns(2)

        if graph_selected(selected_graphs, "Messages by person"):
            with left:
                st.subheader("Messages by person")
                fig, ax = small_fig()
                (
                    analysis_df["Nickname"]
                    .value_counts()
                    .head(15)
                    .sort_values()
                    .plot.barh(ax=ax)
                )
                ax.set_xlabel("Messages")
                ax.set_ylabel("")
                st.pyplot(fig, use_container_width=False)

        if graph_selected(selected_graphs, "Messages by hour"):
            with right:
                st.subheader("Messages by hour")
                fig, ax = small_fig()
                (
                    analysis_df.groupby("Hour")
                    .size()
                    .reindex(range(24), fill_value=0)
                    .plot(ax=ax)
                )
                ax.set_xlabel("Hour")
                ax.set_ylabel("Messages")
                st.pyplot(fig, use_container_width=False)

        if graph_selected(selected_graphs, "Messages over time"):
            st.subheader("Messages over time")
            fig, ax = small_fig(7, 3)
            analysis_df.groupby("DateOnly").size().plot(ax=ax)
            ax.set_xlabel("Date")
            ax.set_ylabel("Messages")
            st.pyplot(fig, use_container_width=False)

        if graph_selected(selected_graphs, "Average sentiment by person"):
            st.subheader("Average sentiment by person")
            fig, ax = small_fig(7, 3)
            (
                analysis_df.groupby("Nickname")["Sentiment"]
                .mean()
                .sort_values()
                .tail(20)
                .plot.barh(ax=ax)
            )
            ax.set_xlabel("Average sentiment")
            ax.set_ylabel("")
            st.pyplot(fig, use_container_width=False)

        if graph_selected(selected_graphs, "Top words"):
            st.subheader("Top words")
            st.dataframe(
                top_words(analysis_df, n=30),
                use_container_width=True,
                hide_index=True,
            )

        if graph_selected(selected_graphs, "Top emojis"):
            st.subheader("Top emojis")
            st.dataframe(
                top_emojis(analysis_df, n=30),
                use_container_width=True,
                hide_index=True,
            )

        if graph_selected(selected_graphs, "Word cloud"):
            st.subheader("Word cloud")
            show_wordcloud(
                " ".join(analysis_df["Text"].fillna("").astype(str)),
                "Overall word cloud",
            )


# =====================================================
# TAB 4 - PERSON GRAPHS
# =====================================================

with tab4:
    st.header("Personalised graphs")

    df = st.session_state.df

    if df.empty:
        st.info("Upload chats first.")
    else:
        filtered, selected_chat = get_filtered_df(df, key_prefix="person_graphs")
        analysis_df = choose_filtered_or_all(
            filtered,
            df,
            selected_chat,
            key_prefix="person_graphs",
        )

        people = sorted(analysis_df["Nickname"].dropna().astype(str).unique())

        if not people:
            st.info("No people found in the selected data.")
        else:
            selected_person = st.selectbox(
                "Choose a person",
                people,
                key="person_graph_person_selector",
            )

            person_df = analysis_df[analysis_df["Nickname"].astype(str) == selected_person]

            person_graph_options = [
                "Messages by hour",
                "Messages over time",
                "Sentiment over time",
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
                    st.pyplot(fig, use_container_width=False)

            if graph_selected(selected_person_graphs, "Messages over time"):
                with right:
                    st.subheader("Messages over time")
                    fig, ax = small_fig()
                    person_df.groupby("DateOnly").size().plot(ax=ax)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Messages")
                    st.pyplot(fig, use_container_width=False)

            if graph_selected(selected_person_graphs, "Sentiment over time"):
                st.subheader("Sentiment over time")
                fig, ax = small_fig(7, 3)
                person_df.groupby("DateOnly")["Sentiment"].mean().plot(ax=ax)
                ax.set_xlabel("Date")
                ax.set_ylabel("Average sentiment")
                st.pyplot(fig, use_container_width=False)

            if graph_selected(selected_person_graphs, "Top words"):
                st.subheader("Top words")
                st.dataframe(
                    top_words(person_df, n=30),
                    use_container_width=True,
                    hide_index=True,
                )

            if graph_selected(selected_person_graphs, "Top emojis"):
                st.subheader("Top emojis")
                st.dataframe(
                    top_emojis(person_df, n=30),
                    use_container_width=True,
                    hide_index=True,
                )

            if graph_selected(selected_person_graphs, "Word cloud"):
                st.subheader("Word cloud")
                show_wordcloud(
                    " ".join(person_df["Text"].fillna("").astype(str)),
                    f"Word cloud for {selected_person}",
                )


# =====================================================
# TAB 5 - SPECIFIC STATS
# =====================================================

with tab5:
    st.header("Specific stats")

    df = st.session_state.df

    if df.empty:
        st.info("Upload chats first.")
    else:
        filtered, selected_chat = get_filtered_df(df, key_prefix="specific_stats")
        analysis_df = choose_filtered_or_all(
            filtered,
            df,
            selected_chat,
            key_prefix="specific_stats",
        )

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
                count = count_cooccurrence(analysis_df, word1, word2)
                st.metric("Messages containing both", count)

        with st.expander("How many times was a word used, and by who?"):
            target_word = st.text_input(
                "Word to analyse",
                key="word_mentions_input",
            )

            if st.button("Analyse word", key="analyse_word_button"):
                result = word_mentions(analysis_df, target_word)

                st.metric("Total mentions", result["total_mentions"])

                st.dataframe(
                    pd.DataFrame(
                        result["mentions_by_player"].items(),
                        columns=["Person", "Mentions"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("How many times did one person use a word?"):
            people = sorted(analysis_df["Nickname"].dropna().astype(str).unique())

            if not people:
                st.info("No people found in the selected data.")
            else:
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
                        analysis_df,
                        selected_person,
                        selected_word,
                    )

                    st.metric("Mentions", count)

        with st.expander("How many times was an exact phrase used?"):
            phrase = st.text_input(
                "Exact phrase",
                key="exact_phrase_input",
            )

            if st.button("Count phrase", key="count_phrase_button"):
                count = count_phrase_mentions(analysis_df, phrase)
                st.metric("Phrase mentions", count)
