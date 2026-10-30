"""TALAD: This American Life Audio Dataset Downloader & Management Toolkit."""

from .catalog import CatalogManager
from .config import AudioConfig, DownloadConfig, EpisodeMetadata, SplitReport, setup_logger
from .converter import AudioConverter
from .downloader import EpisodeDownloader, TALDatasetDownloader, TALDownAudio
from .manifest import DatasetInspector, ManifestExporter

__version__ = "2.0.0"

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
    "__version__",
    "setup_logger",
]
