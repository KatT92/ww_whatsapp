"""
Cache and chat-loading helpers for the WhatsApp Streamlit dashboard.

The permanent cache stores original uploaded .txt files plus metadata.
Processed dataframes are rebuilt from the original txt files when loaded.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
import os
import stat
import time
import pandas as pd

from parse_data import (
    DEFAULT_WHATSAPP_PATTERN,
    parse_whatsapp_text,
    prepare_dataframe,
)


CACHE_DIR = Path("cached_chat_uploads")
CACHE_DIR.mkdir(exist_ok=True)


def safe_cache_name(name: str) -> str:
    """Make a safe folder/file name for a cached chat set."""
    name = str(name).strip()
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
    return name.strip("_") or "chat_cache"


def list_cached_chat_sets() -> list[str]:
    """List cached chat-set folders."""
    return sorted(
        p.name
        for p in CACHE_DIR.iterdir()
        if p.is_dir() and (p / "metadata.json").exists()
    )


def save_uploaded_txt_cache(
    cache_name: str,
    uploaded_files,
    source_mapping: dict[str, str],
    pattern: str,
) -> str:
    """
    Save the original uploaded .txt files plus metadata.

    This deliberately does not save the processed dataframe.
    Cached chat sets are re-parsed from the original text files when loaded.
    """
    base_name = safe_cache_name(cache_name)
    cache_path = CACHE_DIR / base_name

    safe_name = base_name


    counter = 2
    while cache_path.exists():
        safe_name = f"{base_name}_{counter}"
        cache_path = CACHE_DIR / safe_name
        counter += 1

    files_path = cache_path / "files"
    files_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for i, uploaded_file in enumerate(uploaded_files):
        original_name = uploaded_file.name
        saved_name = f"{i:03d}_{safe_cache_name(original_name)}.txt"
        saved_path = files_path / saved_name

        saved_path.write_bytes(uploaded_file.getvalue())

        saved_files.append(
            {
                "original_name": original_name,
                "saved_name": saved_name,
                "source_name": source_mapping.get(original_name, original_name),
            }
        )

    metadata = {
        "cache_name": safe_name,
        "pattern": pattern,
        "files": saved_files,
    }

    (cache_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return safe_name


def load_cached_txt_cache(cache_name: str):
    """
    Load original cached .txt files and metadata.

    Returns:
    - file_data: tuple of (saved filename, file bytes)
    - source_mapping: dict of saved filename -> source/chat name
    - pattern: regex pattern
    """
    safe_name = safe_cache_name(cache_name)
    cache_path = CACHE_DIR / safe_name
    metadata_path = cache_path / "metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    file_data = []
    source_mapping = {}

    for item in metadata["files"]:
        saved_name = item["saved_name"]
        file_path = cache_path / "files" / saved_name

        file_data.append((saved_name, file_path.read_bytes()))
        source_mapping[saved_name] = item["source_name"]

    return (
        tuple(file_data),
        source_mapping,
        metadata.get("pattern", DEFAULT_WHATSAPP_PATTERN),
    )


def _handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def delete_cached_chat_set(cache_name: str) -> bool:
    safe_name = safe_cache_name(cache_name)
    cache_path = CACHE_DIR / safe_name

    if not cache_path.exists():
        return True

    deleted_path = CACHE_DIR / f"__deleted__{safe_name}_{int(time.time())}"

    try:
        cache_path.rename(deleted_path)
        shutil.rmtree(deleted_path, onerror=_handle_remove_readonly)
        return True

    except PermissionError:
        return False


def source_mapping_items(source_mapping: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Make source mapping hashable for st.cache_data."""
    return tuple(sorted(source_mapping.items()))


def load_chats_from_bytes(
    file_data: tuple[tuple[str, bytes], ...],
    source_mapping_items: tuple[tuple[str, str], ...],
    pattern: str = DEFAULT_WHATSAPP_PATTERN,
) -> pd.DataFrame:
    """
    Parse and merge all uploaded/cached txt files.

    Parameters:
    - file_data: tuple of (filename, file bytes)
    - source_mapping_items: tuple of (filename, source/chat name)
    - pattern: WhatsApp regex pattern

    Returns:
    - prepared merged dataframe with Source preserved for chat filtering
    """
    source_mapping = dict(source_mapping_items)
    dfs = []

    for filename, file_bytes in file_data:
        text = file_bytes.decode("utf-8", errors="ignore")

        parsed = parse_whatsapp_text(
            text=text,
            source_name=source_mapping.get(filename, filename),
            pattern=pattern,
        )

        if not parsed.empty:
            dfs.append(parsed)

    if not dfs:
        empty = pd.DataFrame(columns=["Date", "Time", "WA_Name", "Text", "Source"])
        return prepare_dataframe(empty)

    merged = pd.concat(dfs, ignore_index=True)
    return prepare_dataframe(merged)
