"""Configuration models, constants, and logging utilities for TALAD."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = PACKAGE_ROOT / "source"
DEFAULT_SPLIT_DIR = PACKAGE_ROOT / "train_test_valid_split"

UPSTREAM_ARCHIVE_CSV_URL = (
    "https://raw.githubusercontent.com/dcadata/this-american-life-archive/master/data/transformed.csv"
)
OFFICIAL_EPISODE_URL_TEMPLATE = "https://www.thisamericanlife.org/episode/{ep_number}"

VALID_BASE_SPLITS = ("train", "valid", "test")
MIN_VALID_AUDIO_BYTES = 10 * 1024  # 10 KB minimum sanity threshold


def setup_logger(name: str = "talad", verbose: bool = False) -> logging.Logger:
    """Configure and return a structured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s | %(levelname)-8s | %(message)s" if verbose else "%(levelname)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger


@dataclass(frozen=True)
class AudioConfig:
    """Configuration for audio transcoding and post-processing."""

    convert_to_wav: bool = False
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    keep_mp3: bool = False
    output_format: str = "wav"

    def __post_init__(self) -> None:
        if int(self.sample_rate) <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if int(self.channels) not in (1, 2):
            raise ValueError(f"channels must be 1 (mono) or 2 (stereo), got {self.channels}")
        if int(self.bit_depth) not in (8, 16, 24, 32):
            raise ValueError(f"bit_depth must be one of (8, 16, 24, 32), got {self.bit_depth}")


@dataclass(frozen=True)
class DownloadConfig:
    """Configuration for dataset downloading and network behavior."""

    download_dir: Path = field(default_factory=lambda: Path("./data").resolve())
    data_parts: List[str] = field(default_factory=lambda: ["test"])
    episodes: Optional[List[str]] = None
    max_workers: int = 4
    retries: int = 3
    timeout: int = 30
    backend: str = "auto"  # "auto", "urllib", or "wget"
    force: bool = False
    dry_run: bool = False
    num_train_parts: int = 4
    update_catalog: bool = False
    export_manifest: Optional[Path] = None
    manifest_format: str = "jsonl"

    def __post_init__(self) -> None:
        object.__setattr__(self, "download_dir", Path(self.download_dir).resolve())
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")
        if self.backend not in ("auto", "urllib", "wget"):
            raise ValueError("backend must be one of: 'auto', 'urllib', 'wget'")


@dataclass
class EpisodeMetadata:
    """Metadata representation for a single This American Life episode."""

    episode_id: str
    title: str = ""
    download_url: str = ""
    description: str = ""
    pubdate: str = ""
    episode_url: str = ""
    split: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Serialize episode metadata to a dictionary."""
        return {
            "episode_id": self.episode_id,
            "title": self.title,
            "download_url": self.download_url,
            "description": self.description,
            "pubdate": self.pubdate,
            "episode_url": self.episode_url,
            "split": self.split,
        }


@dataclass
class SplitReport:
    """Summary statistics for a dataset split."""

    split_name: str
    target_dir: Path
    total_expected: int
    downloaded_count: int
    missing_episodes: List[str]
    corrupted_episodes: List[str] = field(default_factory=list)
    total_size_bytes: int = 0

    @property
    def missing_count(self) -> int:
        return len(self.missing_episodes)

    @property
    def completion_rate(self) -> float:
        if self.total_expected == 0:
            return 1.0
        return self.downloaded_count / self.total_expected

    @property
    def missing_percentage(self) -> float:
        return (1.0 - self.completion_rate) * 100.0

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)
