"""Catalog management, split resolution, and multi-tier URL discovery."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    DEFAULT_SOURCE_DIR,
    DEFAULT_SPLIT_DIR,
    OFFICIAL_EPISODE_URL_TEMPLATE,
    UPSTREAM_ARCHIVE_CSV_URL,
    VALID_BASE_SPLITS,
    EpisodeMetadata,
)

logger = logging.getLogger("talad")


class CatalogManager:
    """Manages episode splits, archive metadata, and multi-tier audio URL resolution."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TALAD/2.0"

    def __init__(
        self,
        source_dir: Path = DEFAULT_SOURCE_DIR,
        split_dir: Path = DEFAULT_SPLIT_DIR,
    ) -> None:
        self.source_dir = Path(source_dir).resolve()
        self.split_dir = Path(split_dir).resolve()
        self._catalog: Optional[Dict[str, EpisodeMetadata]] = None
        self._legacy_catalog: Optional[Dict[str, EpisodeMetadata]] = None
        self._split_map: Optional[Dict[str, str]] = None

    def get_split_episodes(self, split_spec: str, num_train_parts: int = 4) -> Tuple[str, List[str]]:
        """Resolve a split specification into (base_split_name, episode_list).

        Supported split_spec values:
        - 'train', 'valid', 'test'
        - 'train_part_1' .. 'train_part_N'
        - 'all' (returns all episodes across train, valid, test)
        """
        spec = split_spec.strip().lower()
        if spec == "all":
            all_eps: List[str] = []
            for s in VALID_BASE_SPLITS:
                all_eps.extend(self._read_split_file(s))
            return "all", all_eps

        part_match = re.match(r"^(train|valid|test)_part_(\d+)(?:_of_(\d+))?$", spec)
        if part_match:
            base_split = part_match.group(1)
            part_idx = int(part_match.group(2))
            total_parts = int(part_match.group(3)) if part_match.group(3) else num_train_parts
            if part_idx < 1 or part_idx > total_parts:
                raise ValueError(
                    f"Invalid partition index {part_idx} in '{split_spec}' (expected 1..{total_parts})"
                )
            episodes = self._read_split_file(base_split)
            chunk_size = int(math.ceil(len(episodes) / total_parts))
            start = (part_idx - 1) * chunk_size
            end = part_idx * chunk_size
            return base_split, episodes[start:end]

        if spec in VALID_BASE_SPLITS:
            return spec, self._read_split_file(spec)

        raise ValueError(
            f"Unknown split specification '{split_spec}'. "
            "Valid options: train, valid, test, all, or train_part_<1..4>."
        )

    def _read_split_file(self, base_split: str) -> List[str]:
        split_file = self.split_dir / f"{base_split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def get_episode_split_map(self) -> Dict[str, str]:
        """Return a mapping of episode_id -> split ('train', 'valid', 'test')."""
        if self._split_map is not None:
            return self._split_map
        self._split_map = {}
        for split_name in VALID_BASE_SPLITS:
            try:
                for ep in self._read_split_file(split_name):
                    self._split_map[ep] = split_name
            except FileNotFoundError:
                continue
        return self._split_map

    def update_archive_csv(self, timeout: int = 20) -> bool:
        """Download the latest transformed.csv catalog from the upstream repository."""
        self.source_dir.mkdir(parents=True, exist_ok=True)
        csv_file = self.source_dir / "transformed.csv"
        logger.info("Fetching latest episode catalog from %s...", UPSTREAM_ARCHIVE_CSV_URL)
        try:
            req = urllib.request.Request(
                UPSTREAM_ARCHIVE_CSV_URL,
                headers={"User-Agent": self.USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read()
                if len(content) > 1024:
                    csv_file.write_bytes(content)
                    self._catalog = None  # invalidate cache
                    logger.info("Successfully updated catalog (%d bytes).", len(content))
                    return True
        except Exception as exc:
            logger.warning("Could not download updated archive CSV: %s", exc)
        return False

    def load_catalog(self, force_update: bool = False) -> Dict[str, EpisodeMetadata]:
        """Load the primary and fallback CSV catalogs into memory."""
        if force_update:
            self.update_archive_csv()

        if self._catalog is not None:
            return self._catalog

        self._catalog = {}
        self.source_dir.mkdir(parents=True, exist_ok=True)
        csv_file = self.source_dir / "transformed.csv"

        if not csv_file.is_file():
            self.update_archive_csv()

        split_map = self.get_episode_split_map()

        if csv_file.is_file():
            self._parse_csv_into(csv_file, self._catalog, split_map)

        # Also load legacy TALArchive.csv as fallback for metadata/links
        self._legacy_catalog = {}
        legacy_file = self.source_dir / "TALArchive.csv"
        if legacy_file.is_file():
            self._parse_csv_into(legacy_file, self._legacy_catalog, split_map)
            for ep_id, meta in self._legacy_catalog.items():
                if ep_id not in self._catalog:
                    self._catalog[ep_id] = meta

        return self._catalog

    @staticmethod
    def _parse_csv_into(
        csv_file: Path,
        target: Dict[str, EpisodeMetadata],
        split_map: Dict[str, str],
    ) -> None:
        try:
            with open(csv_file, mode="r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    num = str(row.get("num") or row.get("number") or "").strip()
                    if not num:
                        continue
                    title = str(row.get("title") or row.get("name") or "").strip()
                    download_url = str(row.get("download_url") or row.get("url") or "").strip()
                    description = str(row.get("description") or "").strip()
                    pubdate = str(row.get("pubdate") or row.get("airdate") or "").strip()
                    episode_url = str(
                        row.get("full_url") or row.get("desc_url") or row.get("url") or ""
                    ).strip()
                    if num not in target:
                        target[num] = EpisodeMetadata(
                            episode_id=num,
                            title=title,
                            download_url=download_url,
                            description=description,
                            pubdate=pubdate,
                            episode_url=episode_url,
                            split=split_map.get(num),
                        )
        except Exception as exc:
            logger.warning("Error parsing catalog CSV %s: %s", csv_file, exc)

    def get_metadata(self, ep_number: str) -> EpisodeMetadata:
        """Retrieve metadata for an episode, returning a stub if not in catalog."""
        catalog = self.load_catalog()
        ep_key = str(ep_number).strip()
        if ep_key in catalog:
            return catalog[ep_key]
        split_map = self.get_episode_split_map()
        return EpisodeMetadata(
            episode_id=ep_key,
            episode_url=OFFICIAL_EPISODE_URL_TEMPLATE.format(ep_number=ep_key),
            split=split_map.get(ep_key),
        )

    def search(
        self,
        query: Optional[str] = None,
        split: Optional[str] = None,
        limit: int = 25,
    ) -> List[EpisodeMetadata]:
        """Search episodes in the catalog by title, description, or episode number."""
        catalog = self.load_catalog()
        results: List[EpisodeMetadata] = []
        q = query.strip().lower() if query else ""

        for meta in catalog.values():
            if split and meta.split != split:
                continue
            if q:
                haystack = f"{meta.episode_id} {meta.title} {meta.description}".lower()
                if q not in haystack:
                    continue
            results.append(meta)

        results.sort(key=lambda m: int(m.episode_id) if m.episode_id.isdigit() else 999999)
        return results[:limit] if limit > 0 else results

    def resolve_candidate_urls(self, ep_number: str, timeout: int = 10) -> List[str]:
        """Return an ordered list of candidate audio URLs using 3-tier resolution.

        Tier 1: Maintained archive catalog (transformed.csv)
        Tier 2: Official This American Life episode page embedded JSON
        Tier 3: Legacy archive catalog (TALArchive.csv)
        """
        ep_key = str(ep_number).strip()
        catalog = self.load_catalog()
        candidates: List[str] = []

        # Tier 1: Primary archive CSV
        primary = catalog.get(ep_key)
        if primary and primary.download_url and primary.download_url != "False":
            candidates.append(primary.download_url)

        # Tier 2: Official web scraper
        live_url = self.scrape_official_audio_url(ep_key, timeout=timeout)
        if live_url and live_url not in candidates:
            candidates.append(live_url)

        # Tier 3: Legacy archive CSV
        if self._legacy_catalog and ep_key in self._legacy_catalog:
            legacy_url = self._legacy_catalog[ep_key].download_url
            if legacy_url and legacy_url != "False" and legacy_url not in candidates:
                candidates.append(legacy_url)

        return candidates

    def link_from_archive_csv(self, ep_number: str) -> Optional[str]:
        """Return the primary catalog URL for an episode (if present)."""
        catalog = self.load_catalog()
        meta = catalog.get(str(ep_number).strip())
        if meta and meta.download_url and meta.download_url != "False":
            return meta.download_url
        return None

    @classmethod
    def scrape_official_audio_url(cls, ep_number: str, timeout: int = 10) -> Optional[str]:
        """Scrape the canonical This American Life episode page for its playlist JSON audio link."""
        url = OFFICIAL_EPISODE_URL_TEMPLATE.format(ep_number=ep_number)
        req = urllib.request.Request(url, headers={"User-Agent": cls.USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                match = re.search(
                    r'<script\s+id="playlist-data"[^>]*>(.*?)</script>',
                    html,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                if match:
                    data = json.loads(match.group(1))
                    audio_url = data.get("archive") or data.get("audio")
                    if audio_url:
                        return str(audio_url).strip()
        except Exception:
            pass
        return None
