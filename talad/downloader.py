"""Core episode downloader with multi-threaded execution, resume support, and fallbacks."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .catalog import CatalogManager
from .config import (
    DEFAULT_SOURCE_DIR,
    DEFAULT_SPLIT_DIR,
    MIN_VALID_AUDIO_BYTES,
    UPSTREAM_ARCHIVE_CSV_URL,
    AudioConfig,
    DownloadConfig,
    SplitReport,
)
from .converter import AudioConverter
from .manifest import DatasetInspector, ManifestExporter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):  # type: ignore[misc]
        return iterable

logger = logging.getLogger("talad")


class EpisodeDownloader:
    """Handles resilient single-episode downloading with retries and atomic writes."""

    CHUNK_SIZE = 64 * 1024  # 64 KB chunks
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TALAD/2.0"

    def __init__(
        self,
        catalog: CatalogManager,
        backend: str = "auto",
        retries: int = 3,
        timeout: int = 30,
    ) -> None:
        self.catalog = catalog
        self.backend = backend
        self.retries = retries
        self.timeout = timeout
        self._has_wget = shutil.which("wget") is not None

    def download_url(self, url: str, target_file: Path) -> bool:
        """Download a single URL to target_file using the configured backend."""
        if not url or url == "False":
            return False

        target_file = Path(target_file)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        part_file = target_file.with_suffix(target_file.suffix + ".part")

        use_wget = self.backend == "wget" or (self.backend == "auto" and False)
        if use_wget and self._has_wget:
            return self._download_with_wget(url, target_file, part_file)

        # Default to native Python HTTP streaming with fallback to wget if available
        if self._download_with_urllib(url, target_file, part_file):
            return True
        if self.backend == "auto" and self._has_wget:
            return self._download_with_wget(url, target_file, part_file)
        return False

    def _download_with_urllib(self, url: str, target_file: Path, part_file: Path) -> bool:
        for attempt in range(self.retries + 1):
            try:
                existing_bytes = part_file.stat().st_size if part_file.is_file() else 0
                headers = {"User-Agent": self.USER_AGENT}
                if existing_bytes > 0:
                    headers["Range"] = f"bytes={existing_bytes}-"

                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = getattr(resp, "status", 200)
                    mode = "ab" if (existing_bytes > 0 and status == 206) else "wb"
                    with open(part_file, mode) as f:
                        while True:
                            chunk = resp.read(self.CHUNK_SIZE)
                            if not chunk:
                                break
                            f.write(chunk)

                if part_file.is_file() and part_file.stat().st_size > MIN_VALID_AUDIO_BYTES:
                    part_file.replace(target_file)
                    return True
                part_file.unlink(missing_ok=True)
            except urllib.error.HTTPError as exc:
                if exc.code == 416:  # Range Not Satisfiable -> reset partial file
                    part_file.unlink(missing_ok=True)
                elif exc.code == 404:
                    break  # Do not retry 404 on the same URL
            except Exception as exc:
                logger.debug("Download attempt %d failed for %s: %s", attempt + 1, url, exc)

            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 8))

        return False

    def _download_with_wget(self, url: str, target_file: Path, part_file: Path) -> bool:
        try:
            cmd = [
                "wget",
                "-q",
                "-c",
                "--tries",
                str(max(1, self.retries)),
                "--timeout",
                str(self.timeout),
                "-O",
                str(part_file),
                url,
            ]
            proc = subprocess.run(cmd, check=False)
            if proc.returncode == 0 and part_file.is_file():
                if part_file.stat().st_size > MIN_VALID_AUDIO_BYTES:
                    part_file.replace(target_file)
                    return True
            part_file.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("wget error for %s: %s", url, exc)
        return False

    def download_episode(
        self,
        ep_number: str,
        destination_dir: Path,
        force: bool = False,
        dry_run: bool = False,
    ) -> Tuple[str, bool, Optional[str]]:
        """Resolve and download a single episode into destination_dir.

        Returns (ep_number, success, resolved_url_or_status).
        """
        ep_key = str(ep_number).strip()
        destination_dir = Path(destination_dir)
        mp3_file = destination_dir / f"{ep_key}.mp3"
        wav_file = destination_dir / f"{ep_key}.wav"
        flac_file = destination_dir / f"{ep_key}.flac"

        if not force:
            for existing in (mp3_file, wav_file, flac_file):
                if existing.is_file() and existing.stat().st_size > MIN_VALID_AUDIO_BYTES:
                    return ep_key, True, "cached"

        candidates = self.catalog.resolve_candidate_urls(ep_key, timeout=min(self.timeout, 12))
        if dry_run:
            best_url = candidates[0] if candidates else None
            return ep_key, best_url is not None, best_url

        for url in candidates:
            if self.download_url(url, mp3_file):
                return ep_key, True, url

        return ep_key, False, None


class TALDatasetDownloader:
    """High-level orchestrator for downloading, converting, and verifying TALAD splits."""

    transcript_dir_path = "/kaggle/input/this-american-life-podcast-transcriptsalignments/"
    SOURCE_CSV_URL = UPSTREAM_ARCHIVE_CSV_URL

    def __init__(
        self,
        data_parts: Optional[Union[Sequence[str], str]] = None,
        DOWNLOAD_DIR: Union[str, Path] = "./data",
        *,
        download_config: Optional[DownloadConfig] = None,
        audio_config: Optional[AudioConfig] = None,
        source_dir: Optional[Path] = None,
        split_dir: Optional[Path] = None,
    ) -> None:
        if isinstance(data_parts, str):
            parts_list = [data_parts]
        elif data_parts is not None:
            parts_list = list(data_parts)
        elif download_config is not None:
            parts_list = list(download_config.data_parts)
        else:
            parts_list = ["test"]

        self.download_config = download_config or DownloadConfig(
            download_dir=Path(DOWNLOAD_DIR).resolve(),
            data_parts=parts_list,
        )
        self.audio_config = audio_config or AudioConfig()
        self.data_parts = self.download_config.data_parts
        self.download_dir_path = self.download_config.download_dir

        self.working_dir = Path.cwd().resolve()
        self.source_dir = Path(source_dir) if source_dir else DEFAULT_SOURCE_DIR
        self.split_dir = Path(split_dir) if split_dir else DEFAULT_SPLIT_DIR

        self.catalog = CatalogManager(source_dir=self.source_dir, split_dir=self.split_dir)
        self.episode_downloader = EpisodeDownloader(
            catalog=self.catalog,
            backend=self.download_config.backend,
            retries=self.download_config.retries,
            timeout=self.download_config.timeout,
        )
        self.inspector = DatasetInspector(self.download_dir_path, catalog=self.catalog)
        self.manifest_exporter = ManifestExporter(self.download_dir_path, catalog=self.catalog)

        # Legacy attributes for backward compatibility
        self._archive_map: Optional[Dict[str, str]] = None
        self.mode: str = "test"
        self.list_of_ep: List[str] = []
        self.convert_to_wav: bool = self.audio_config.convert_to_wav
        self.ar: str = str(self.audio_config.sample_rate)
        self.ac: str = str(self.audio_config.channels)
        self.ab: str = str(self.audio_config.bit_depth)

    # ------------------------------------------------------------------
    # Legacy helper methods preserved for backward compatibility
    # ------------------------------------------------------------------
    def _list_episodes(self, mode: str) -> List[str]:
        _, episodes = self.catalog.get_split_episodes(
            mode, num_train_parts=self.download_config.num_train_parts
        )
        return episodes

    def _load_archive_csv(self) -> Dict[str, str]:
        if self._archive_map is None:
            catalog_data = self.catalog.load_catalog()
            self._archive_map = {
                ep_id: meta.download_url
                for ep_id, meta in catalog_data.items()
                if meta.download_url
            }
        return self._archive_map

    def _link_from_archive_csv(self, ep_number: Union[str, int]) -> Optional[str]:
        return self.catalog.link_from_archive_csv(str(ep_number))

    def _link_from_official_web(self, ep_number: Union[str, int]) -> Optional[str]:
        return self.catalog.scrape_official_audio_url(str(ep_number))

    @staticmethod
    def _try_link(link: Optional[str], file_path: Optional[Union[str, Path]] = None) -> int:
        if not link or link == "False":
            return 1
        target = Path(file_path) if file_path else Path("temp_episode.mp3")
        downloader = EpisodeDownloader(CatalogManager())
        return 0 if downloader.download_url(link, target) else 1

    # ------------------------------------------------------------------
    # Core execution pipeline
    # ------------------------------------------------------------------
    def download_episodes_list(
        self,
        episode_list: Sequence[str],
        destination_path: Path,
        label: str = "custom",
    ) -> List[str]:
        """Download a list of episode IDs into destination_path, optionally in parallel."""
        destination_path = Path(destination_path)
        destination_path.mkdir(parents=True, exist_ok=True)
        workers = self.download_config.max_workers
        force = self.download_config.force
        dry_run = self.download_config.dry_run

        action = "Resolving (dry-run)" if dry_run else "Downloading"
        logger.info(
            "%s %s dataset (%d episode(s), workers=%d) -> %s",
            action,
            label,
            len(episode_list),
            workers,
            destination_path,
        )

        failed: List[str] = []
        desc = f"{action} [{label}]"

        if workers <= 1 or len(episode_list) <= 1:
            for ep_num in tqdm(episode_list, desc=desc, unit="ep"):
                ep_id, ok, info = self.episode_downloader.download_episode(
                    ep_num, destination_path, force=force, dry_run=dry_run
                )
                if dry_run:
                    logger.info("  Episode %-4s -> %s", ep_id, info or "UNRESOLVED")
                if not ok:
                    failed.append(ep_id)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        self.episode_downloader.download_episode,
                        ep_num,
                        destination_path,
                        force,
                        dry_run,
                    ): ep_num
                    for ep_num in episode_list
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="ep"):
                    ep_id, ok, info = future.result()
                    if dry_run:
                        logger.info("  Episode %-4s -> %s", ep_id, info or "UNRESOLVED")
                    if not ok:
                        failed.append(ep_id)

        if failed:
            failed_sorted = sorted(failed, key=lambda x: int(x) if x.isdigit() else 999999)
            logger.warning(
                "Could not download %d episode(s) in %s: %s",
                len(failed_sorted),
                label,
                failed_sorted,
            )
        return failed

    def _download_mp3(
        self,
        episode_list: Sequence[str],
        destination_path: Union[str, Path],
        mode: str,
    ) -> None:
        self.download_episodes_list(episode_list, Path(destination_path), label=mode)

    def _convert_mp3_to_wav(self, folder_path: Union[str, Path]) -> None:
        converter = AudioConverter(self.audio_config)
        converter.convert_directory(
            Path(folder_path),
            max_workers=max(1, min(self.download_config.max_workers, 4)),
            desc=f"Converting [{self.mode}]",
        )

    def _missing_data_percentage(self, mode: str) -> SplitReport:
        report = self.inspector.inspect_split(mode, expected_episodes=self.list_of_ep)
        logger.info(
            "Split '%s' status: %d/%d episodes available (%.1f%% complete, %.1f MB, %d%% missing).",
            mode,
            report.downloaded_count,
            report.total_expected,
            report.completion_rate * 100.0,
            report.total_size_mb,
            int(report.missing_percentage),
        )
        if report.missing_episodes:
            logger.info("Missing episodes in '%s': %s", mode, report.missing_episodes)
        return report

    def _download_part(self, mode: str) -> SplitReport:
        base_split, episodes = self.catalog.get_split_episodes(
            mode, num_train_parts=self.download_config.num_train_parts
        )
        self.mode = base_split
        self.list_of_ep = episodes
        dir_path = self.download_dir_path / base_split

        self._download_mp3(self.list_of_ep, dir_path, mode)
        if self.audio_config.convert_to_wav and not self.download_config.dry_run:
            self._convert_mp3_to_wav(dir_path)

        return self._missing_data_percentage(mode)

    def download(
        self,
        convert_to_wav: Optional[bool] = None,
        ar: Optional[Union[str, int]] = None,
        ac: Optional[Union[str, int]] = None,
        ab: Optional[Union[str, int]] = None,
        keep_mp3: Optional[bool] = None,
    ) -> List[SplitReport]:
        """Run the full download (and optional conversion/manifest export) workflow."""
        if any(x is not None for x in (convert_to_wav, ar, ac, ab, keep_mp3)):
            self.audio_config = AudioConfig(
                convert_to_wav=self.audio_config.convert_to_wav
                if convert_to_wav is None
                else bool(convert_to_wav),
                sample_rate=int(ar) if ar is not None else self.audio_config.sample_rate,
                channels=int(ac) if ac is not None else self.audio_config.channels,
                bit_depth=int(ab) if ab is not None else self.audio_config.bit_depth,
                keep_mp3=self.audio_config.keep_mp3 if keep_mp3 is None else bool(keep_mp3),
                output_format=self.audio_config.output_format,
            )
            self.convert_to_wav = self.audio_config.convert_to_wav
            self.ar = str(self.audio_config.sample_rate)
            self.ac = str(self.audio_config.channels)
            self.ab = str(self.audio_config.bit_depth)

        if self.download_config.update_catalog:
            self.catalog.update_archive_csv()

        reports: List[SplitReport] = []

        # If explicit episode numbers were requested, download them into 'custom' (or split dir)
        if self.download_config.episodes:
            custom_eps = [str(e).strip() for e in self.download_config.episodes if str(e).strip()]
            custom_dir = self.download_dir_path / "episodes"
            self.mode = "episodes"
            self.list_of_ep = custom_eps
            self._download_mp3(custom_eps, custom_dir, "episodes")
            if self.audio_config.convert_to_wav and not self.download_config.dry_run:
                self._convert_mp3_to_wav(custom_dir)
        else:
            for part in self.data_parts:
                if part == "all":
                    for split_name in ("test", "valid", "train"):
                        reports.append(self._download_part(split_name))
                else:
                    reports.append(self._download_part(part))

        if reports:
            print(self.inspector.format_report_table(reports))

        if self.download_config.export_manifest and not self.download_config.dry_run:
            self.manifest_exporter.export(
                output_path=self.download_config.export_manifest,
                fmt=self.download_config.manifest_format,
                splits=self.data_parts if "all" not in self.data_parts else None,
            )

        return reports

    def remove_dir(self) -> None:
        """Delete downloaded files and split directories for configured data_parts."""
        for part in self.data_parts:
            base_split = "train" if "part" in part else part
            pth = self.download_dir_path / base_split
            if pth.is_dir():
                for child in pth.glob("*"):
                    if child.is_file():
                        child.unlink()
                try:
                    pth.rmdir()
                except OSError:
                    pass


# Backward-compatible class alias
TALDownAudio = TALDatasetDownloader
