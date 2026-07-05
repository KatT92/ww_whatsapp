"""
Plot helpers for the WhatsApp Streamlit dashboard.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def small_fig(width: int = 6, height: int = 3):
    """Create a compact matplotlib figure."""
    return plt.subplots(figsize=(width, height))


def graph_selected(selected_graphs, graph_name: str) -> bool:
    """Return True if a selected graph should be shown."""
    return "View all" in selected_graphs or graph_name in selected_graphs


def show_wordcloud(text: str, title: str = "Word cloud", st=None):
    """
    Render a word cloud.
    """
    if not text or not text.strip():
        if st is not None:
            st.info("No text available for this word cloud.")
        return None

    wc = WordCloud(width=900, height=450, background_color="white").generate(text)

    fig, ax = small_fig(6, 3)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)

    if st is not None:
        st.pyplot(fig, use_container_width=False)

    return fig
