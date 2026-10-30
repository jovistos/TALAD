#!/usr/bin/env python3
"""Entrypoint and backward-compatible facade for the TALAD audio dataset downloader.

Supports both legacy CLI flags:
    python3 TAL_download_audio.py -p ./data -d test valid -w True

And modern subcommands:
    python3 TAL_download_audio.py download -p ./data -d test --workers 8 --wav
    python3 TAL_download_audio.py status -p ./data --show-missing
    python3 TAL_download_audio.py catalog --query "Pulitzer"
    python3 TAL_download_audio.py manifest -p ./data -o ./data/manifest.jsonl
"""

from __future__ import annotations

from talad import (
    AudioConfig,
    AudioConverter,
    CatalogManager,
    DatasetInspector,
    DownloadConfig,
    EpisodeDownloader,
    EpisodeMetadata,
    ManifestExporter,
    SplitReport,
    TALDatasetDownloader,
    TALDownAudio,
)
from talad.cli import main

__all__ = [
    "AudioConfig",
    "AudioConverter",
    "CatalogManager",
    "DatasetInspector",
    "DownloadConfig",
    "EpisodeDownloader",
    "EpisodeMetadata",
    "ManifestExporter",
    "SplitReport",
    "TALDatasetDownloader",
    "TALDownAudio",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
