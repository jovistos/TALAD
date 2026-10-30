"""Dataset inspection, verification, and ML manifest generation."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

from .catalog import CatalogManager
from .config import MIN_VALID_AUDIO_BYTES, VALID_BASE_SPLITS, SplitReport

logger = logging.getLogger("talad")


class DatasetInspector:
    """Inspects local dataset directories, verifies file integrity, and reports statistics."""

    AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}

    def __init__(self, download_dir: Path, catalog: Optional[CatalogManager] = None) -> None:
        self.download_dir = Path(download_dir).resolve()
        self.catalog = catalog or CatalogManager()

    def inspect_split(
        self,
        split_name: str,
        expected_episodes: Optional[List[str]] = None,
    ) -> SplitReport:
        """Analyze a split directory and return a detailed SplitReport."""
        base_split, default_eps = self.catalog.get_split_episodes(split_name)
        episodes = expected_episodes if expected_episodes is not None else default_eps
        target_dir = self.download_dir / base_split

        existing_valid: set[str] = set()
        corrupted: List[str] = []
        total_bytes = 0

        if target_dir.is_dir():
            for file_path in target_dir.iterdir():
                if not file_path.is_file() or file_path.suffix.lower() not in self.AUDIO_EXTENSIONS:
                    continue
                if file_path.name.endswith(".part") or ".tmp." in file_path.name:
                    continue
                size = file_path.stat().st_size
                ep_id = file_path.stem
                if size <= MIN_VALID_AUDIO_BYTES:
                    corrupted.append(ep_id)
                else:
                    existing_valid.add(ep_id)
                    total_bytes += size

        expected_set = set(episodes)
        downloaded_in_split = existing_valid & expected_set
        missing = sorted(
            list(expected_set - downloaded_in_split),
            key=lambda x: int(x) if x.isdigit() else 999999,
        )

        return SplitReport(
            split_name=split_name,
            target_dir=target_dir,
            total_expected=len(expected_set),
            downloaded_count=len(downloaded_in_split),
            missing_episodes=missing,
            corrupted_episodes=sorted(set(corrupted)),
            total_size_bytes=total_bytes,
        )

    def inspect_all_splits(self) -> List[SplitReport]:
        """Inspect train, valid, and test splits."""
        return [self.inspect_split(split) for split in VALID_BASE_SPLITS]

    @staticmethod
    def format_report_table(reports: List[SplitReport]) -> str:
        """Format a list of SplitReports into a clean ASCII summary table."""
        lines = [
            "",
            "=" * 78,
            f"{'Split':<14} | {'Downloaded':<14} | {'Completion':<12} | {'Size (MB)':<12} | {'Missing':<8}",
            "-" * 78,
        ]
        total_exp = 0
        total_dl = 0
        total_bytes = 0

        for r in reports:
            total_exp += r.total_expected
            total_dl += r.downloaded_count
            total_bytes += r.total_size_bytes
            dl_str = f"{r.downloaded_count}/{r.total_expected}"
            comp_str = f"{r.completion_rate * 100:6.1f}%"
            size_str = f"{r.total_size_mb:9.1f} MB"
            lines.append(
                f"{r.split_name:<14} | {dl_str:<14} | {comp_str:<12} | {size_str:<12} | {r.missing_count:<8}"
            )

        lines.append("-" * 78)
        overall_pct = (total_dl / total_exp * 100.0) if total_exp else 100.0
        total_mb = total_bytes / (1024 * 1024)
        lines.append(
            f"{'TOTAL':<14} | {f'{total_dl}/{total_exp}':<14} | "
            f"{f'{overall_pct:6.1f}%':<12} | {f'{total_mb:9.1f} MB':<12} | {total_exp - total_dl:<8}"
        )
        lines.append("=" * 78)
        return "\n".join(lines)


class ManifestExporter:
    """Exports audio dataset metadata and paths into JSONL, CSV, or JSON manifests."""

    def __init__(self, download_dir: Path, catalog: Optional[CatalogManager] = None) -> None:
        self.download_dir = Path(download_dir).resolve()
        self.catalog = catalog or CatalogManager()
        self._has_ffprobe = shutil.which("ffprobe") is not None

    def probe_duration(self, audio_path: Path) -> Optional[float]:
        """Attempt to determine the duration of an audio file in seconds."""
        if audio_path.suffix.lower() == ".wav":
            try:
                with wave.open(str(audio_path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        return round(frames / float(rate), 3)
            except Exception:
                pass

        if self._has_ffprobe:
            try:
                cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
                if proc.returncode == 0 and proc.stdout.strip():
                    return round(float(proc.stdout.strip()), 3)
            except Exception:
                pass
        return None

    def collect_records(
        self,
        splits: Optional[List[str]] = None,
        include_duration: bool = False,
    ) -> List[Dict[str, Any]]:
        """Scan downloaded splits and build manifest records with catalog metadata."""
        target_splits = splits or list(VALID_BASE_SPLITS)
        records: List[Dict[str, Any]] = []

        for split_spec in target_splits:
            base_split, expected_eps = self.catalog.get_split_episodes(split_spec)
            expected_set = set(expected_eps)
            split_dir = self.download_dir / base_split
            if not split_dir.is_dir():
                continue

            audio_files: Dict[str, Path] = {}
            for ext in (".wav", ".flac", ".ogg", ".mp3"):
                for file_path in split_dir.glob(f"*{ext}"):
                    if file_path.stat().st_size > MIN_VALID_AUDIO_BYTES:
                        audio_files.setdefault(file_path.stem, file_path)

            for ep_id in sorted(audio_files.keys(), key=lambda x: int(x) if x.isdigit() else 999999):
                if ep_id not in expected_set:
                    continue
                file_path = audio_files[ep_id]
                meta = self.catalog.get_metadata(ep_id)
                record: Dict[str, Any] = {
                    "audio_filepath": str(file_path.resolve()),
                    "episode_id": ep_id,
                    "split": base_split,
                    "format": file_path.suffix.lstrip(".").lower(),
                    "size_bytes": file_path.stat().st_size,
                    "title": meta.title,
                    "pubdate": meta.pubdate,
                    "episode_url": meta.episode_url,
                    "description": meta.description,
                }
                if include_duration:
                    record["duration"] = self.probe_duration(file_path)
                records.append(record)

        return records

    def export(
        self,
        output_path: Path,
        fmt: str = "jsonl",
        splits: Optional[List[str]] = None,
        include_duration: bool = False,
    ) -> Path:
        """Write manifest records to disk in jsonl, json, or csv format."""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records = self.collect_records(splits=splits, include_duration=include_duration)
        fmt_clean = fmt.strip().lower()

        if fmt_clean == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        elif fmt_clean == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        elif fmt_clean == "csv":
            fieldnames = [
                "audio_filepath",
                "episode_id",
                "split",
                "format",
                "size_bytes",
                "duration",
                "title",
                "pubdate",
                "episode_url",
                "description",
            ]
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for rec in records:
                    writer.writerow(rec)
        else:
            raise ValueError(f"Unsupported manifest format '{fmt}'. Choose from: jsonl, json, csv.")

        logger.info("Exported %d episode record(s) to %s (%s).", len(records), output_path, fmt_clean)
        return output_path
