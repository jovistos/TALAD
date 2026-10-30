"""Audio transcoding and format conversion utilities."""

from __future__ import annotations

import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from .config import MIN_VALID_AUDIO_BYTES, AudioConfig

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):  # type: ignore[misc]
        return iterable

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

logger = logging.getLogger("talad")


class AudioConverter:
    """Converts downloaded MP3 podcast files to standardized WAV/FLAC audio."""

    BIT_DEPTH_CODEC_MAP = {
        8: "pcm_u8",
        16: "pcm_s16le",
        24: "pcm_s24le",
        32: "pcm_s32le",
    }

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        self.config = config or AudioConfig(convert_to_wav=True)
        self._has_ffmpeg = shutil.which("ffmpeg") is not None

    def is_available(self) -> bool:
        """Check whether either ffmpeg or pydub is available for audio conversion."""
        return self._has_ffmpeg or (AudioSegment is not None)

    def convert_file(self, mp3_path: Path) -> Optional[Path]:
        """Convert a single MP3 file to the configured target audio format.

        Returns the path to the converted file on success, or None on failure.
        """
        mp3_path = Path(mp3_path)
        ext = self.config.output_format.lstrip(".")
        target_path = mp3_path.with_suffix(f".{ext}")

        if target_path.is_file() and target_path.stat().st_size > MIN_VALID_AUDIO_BYTES:
            if not self.config.keep_mp3 and mp3_path.is_file():
                mp3_path.unlink(missing_ok=True)
            return target_path

        temp_target = mp3_path.with_suffix(f".tmp.{ext}")
        temp_target.unlink(missing_ok=True)

        success = False
        if self._has_ffmpeg:
            success = self._convert_via_ffmpeg(mp3_path, temp_target, ext)
        elif AudioSegment is not None:
            success = self._convert_via_pydub(mp3_path, temp_target, ext)
        else:
            logger.warning(
                "Neither ffmpeg nor pydub is available; skipping conversion for %s",
                mp3_path.name,
            )
            return None

        if success and temp_target.is_file() and temp_target.stat().st_size > MIN_VALID_AUDIO_BYTES:
            temp_target.replace(target_path)
            if not self.config.keep_mp3:
                mp3_path.unlink(missing_ok=True)
            return target_path

        temp_target.unlink(missing_ok=True)
        return None

    def _convert_via_ffmpeg(self, mp3_path: Path, output_path: Path, ext: str) -> bool:
        codec = self.BIT_DEPTH_CODEC_MAP.get(int(self.config.bit_depth), "pcm_s16le")
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(mp3_path),
            "-ar",
            str(self.config.sample_rate),
            "-ac",
            str(self.config.channels),
        ]
        if ext == "wav":
            cmd.extend(["-c:a", codec])
        cmd.append(str(output_path))

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                return True
            logger.debug("ffmpeg error on %s: %s", mp3_path.name, proc.stderr.strip())
        except Exception as exc:
            logger.debug("ffmpeg exception on %s: %s", mp3_path.name, exc)

        # Fallback to pydub if available
        if AudioSegment is not None:
            return self._convert_via_pydub(mp3_path, output_path, ext)
        return False

    def _convert_via_pydub(self, mp3_path: Path, output_path: Path, ext: str) -> bool:
        try:
            sound = AudioSegment.from_mp3(str(mp3_path))
            sound.export(
                str(output_path),
                format=ext,
                parameters=[
                    "-ar",
                    str(self.config.sample_rate),
                    "-ac",
                    str(self.config.channels),
                    "-ab",
                    str(self.config.bit_depth),
                ],
            )
            return True
        except Exception as exc:
            logger.error("Error converting %s to %s: %s", mp3_path.name, ext, exc)
            return False

    def convert_directory(
        self,
        folder_path: Path,
        max_workers: int = 2,
        desc: str = "Converting MP3 to WAV",
    ) -> List[Path]:
        """Convert all MP3 files in a directory to the configured target audio format."""
        folder_path = Path(folder_path)
        mp3_files = sorted(folder_path.glob("*.mp3"))
        if not mp3_files:
            return []

        if not self.is_available():
            logger.warning(
                "Neither ffmpeg nor pydub is installed. Skipping audio conversion in %s.",
                folder_path,
            )
            return []

        logger.info("Converting %d MP3 file(s) in %s...", len(mp3_files), folder_path)
        converted: List[Path] = []

        if max_workers <= 1 or len(mp3_files) == 1:
            for mp3_path in tqdm(mp3_files, desc=desc, unit="ep"):
                out = self.convert_file(mp3_path)
                if out:
                    converted.append(out)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(self.convert_file, p): p for p in mp3_files}
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="ep"):
                    out = future.result()
                    if out:
                        converted.append(out)

        return sorted(converted)
