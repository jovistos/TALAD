"""Command-line interface for TALAD with subcommands and legacy flag compatibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .catalog import CatalogManager
from .config import AudioConfig, DownloadConfig, setup_logger
from .converter import AudioConverter
from .downloader import TALDatasetDownloader
from .manifest import DatasetInspector, ManifestExporter

SUBCOMMANDS = {"download", "status", "verify", "catalog", "manifest", "convert", "clean"}


def _str_to_bool(val: object) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "y", "on")


def _add_download_arguments(parser: argparse.ArgumentParser, require_target: bool = False) -> None:
    parser.add_argument(
        "-p",
        "--download_path",
        "--output-dir",
        dest="download_path",
        default="./data",
        required=require_target,
        help="Target directory where dataset splits will be stored (default: ./data)",
    )
    parser.add_argument(
        "-d",
        "--data_parts",
        "--splits",
        dest="data_parts",
        nargs="+",
        default=["test"],
        help=(
            "Dataset splits to download: test, valid, train, all, or "
            "train_part_{i} (e.g. train_part_1 .. train_part_4)"
        ),
    )
    parser.add_argument(
        "-e",
        "--episodes",
        nargs="+",
        default=None,
        help="Download specific episode number(s) instead of full splits (e.g. -e 1 186 704)",
    )
    parser.add_argument(
        "-w",
        "--convert_to_wav",
        "--wav",
        dest="convert_to_wav",
        nargs="?",
        const="True",
        default="False",
        help="Convert downloaded MP3 files to WAV (True/False or flag)",
    )
    parser.add_argument(
        "-ar",
        "--sample_rate",
        dest="sample_rate",
        type=int,
        default=16000,
        help="Target audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "-ac",
        "--channels",
        dest="channels",
        type=int,
        default=1,
        help="Target number of audio channels: 1=mono, 2=stereo (default: 1)",
    )
    parser.add_argument(
        "-ab",
        "--bit_depth",
        dest="bit_depth",
        type=int,
        default=16,
        help="Target PCM bit depth: 8, 16, 24, 32 (default: 16)",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["wav", "flac", "ogg"],
        default="wav",
        help="Target audio format when conversion is enabled (default: wav)",
    )
    parser.add_argument(
        "--keep-mp3",
        action="store_true",
        help="Retain original MP3 files after converting to WAV/FLAC",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent download/conversion threads (default: 4)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts per URL on network failure (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Socket timeout in seconds per HTTP request (default: 30)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "urllib", "wget"],
        default="auto",
        help="Download HTTP backend (default: auto)",
    )
    parser.add_argument(
        "--num-train-parts",
        type=int,
        default=4,
        help="Total number of partitions when using train_part_i (default: 4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-downloading episodes even if local files already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print episode audio URLs without downloading files",
    )
    parser.add_argument(
        "--update-catalog",
        action="store_true",
        help="Refresh source/transformed.csv from upstream GitHub before downloading",
    )
    parser.add_argument(
        "--export-manifest",
        type=Path,
        default=None,
        help="Optional path to export a dataset manifest after downloading (e.g. manifest.jsonl)",
    )
    parser.add_argument(
        "--manifest-format",
        choices=["jsonl", "json", "csv"],
        default="jsonl",
        help="Format for --export-manifest (default: jsonl)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )


def _run_download(args: argparse.Namespace) -> int:
    setup_logger(verbose=getattr(args, "verbose", False))
    convert_flag = _str_to_bool(args.convert_to_wav)

    audio_cfg = AudioConfig(
        convert_to_wav=convert_flag,
        sample_rate=args.sample_rate,
        channels=args.channels,
        bit_depth=args.bit_depth,
        keep_mp3=args.keep_mp3,
        output_format=args.output_format,
    )
    dl_cfg = DownloadConfig(
        download_dir=Path(args.download_path).resolve(),
        data_parts=args.data_parts,
        episodes=args.episodes,
        max_workers=args.workers,
        retries=args.retries,
        timeout=args.timeout,
        backend=args.backend,
        force=args.force,
        dry_run=args.dry_run,
        num_train_parts=args.num_train_parts,
        update_catalog=args.update_catalog,
        export_manifest=args.export_manifest,
        manifest_format=args.manifest_format,
    )

    downloader = TALDatasetDownloader(
        data_parts=dl_cfg.data_parts,
        DOWNLOAD_DIR=dl_cfg.download_dir,
        download_config=dl_cfg,
        audio_config=audio_cfg,
    )
    downloader.download()
    return 0


def _run_status(args: argparse.Namespace) -> int:
    setup_logger(verbose=getattr(args, "verbose", False))
    inspector = DatasetInspector(Path(args.download_path))
    splits = args.splits or ["test", "valid", "train"]
    reports = [inspector.inspect_split(s) for s in splits]
    print(inspector.format_report_table(reports))

    if args.show_missing:
        for r in reports:
            if r.missing_episodes:
                print(f"\n[{r.split_name}] Missing episodes ({r.missing_count}):")
                print(", ".join(r.missing_episodes))
            if r.corrupted_episodes:
                print(f"\n[{r.split_name}] Corrupted/incomplete files ({len(r.corrupted_episodes)}):")
                print(", ".join(r.corrupted_episodes))
    return 0


def _run_catalog(args: argparse.Namespace) -> int:
    setup_logger(verbose=getattr(args, "verbose", False))
    catalog = CatalogManager()

    if args.update:
        catalog.update_archive_csv()

    if args.episode:
        meta = catalog.get_metadata(args.episode)
        urls = catalog.resolve_candidate_urls(args.episode)
        print(f"\nEpisode #{meta.episode_id}: {meta.title or '(Untitled)'}")
        print("-" * 60)
        print(f"Split       : {meta.split or 'unassigned'}")
        print(f"Air Date    : {meta.pubdate or 'N/A'}")
        print(f"Web Page    : {meta.episode_url or 'N/A'}")
        print(f"Primary URL : {meta.download_url or 'N/A'}")
        if urls:
            print(f"Resolved URL: {urls[0]}")
        if meta.description:
            print(f"\nDescription :\n  {meta.description}")
        return 0

    results = catalog.search(query=args.query, split=args.split, limit=args.limit)
    print(f"\nFound {len(results)} matching episode(s):")
    print("=" * 88)
    print(f"{'Ep #':<6} | {'Split':<7} | {'Air Date':<16} | {'Title'}")
    print("-" * 88)
    for m in results:
        short_date = m.pubdate[:16] if m.pubdate else "N/A"
        title = (m.title[:50] + "...") if len(m.title) > 53 else m.title
        print(f"{m.episode_id:<6} | {(m.split or '-'):<7} | {short_date:<16} | {title}")
    print("=" * 88)
    return 0


def _run_manifest(args: argparse.Namespace) -> int:
    setup_logger(verbose=getattr(args, "verbose", False))
    exporter = ManifestExporter(Path(args.download_path))
    exporter.export(
        output_path=Path(args.output),
        fmt=args.format,
        splits=args.splits,
        include_duration=args.probe_duration,
    )
    return 0


def _run_convert(args: argparse.Namespace) -> int:
    setup_logger(verbose=getattr(args, "verbose", False))
    cfg = AudioConfig(
        convert_to_wav=True,
        sample_rate=args.sample_rate,
        channels=args.channels,
        bit_depth=args.bit_depth,
        keep_mp3=args.keep_mp3,
        output_format=args.format,
    )
    converter = AudioConverter(cfg)
    target = Path(args.folder).resolve()
    converter.convert_directory(target, max_workers=args.workers)
    return 0


def _run_clean(args: argparse.Namespace) -> int:
    setup_logger(verbose=getattr(args, "verbose", False))
    downloader = TALDatasetDownloader(data_parts=args.splits, DOWNLOAD_DIR=args.download_path)
    downloader.remove_dir()
    print(f"Cleaned splits {args.splits} under {Path(args.download_path).resolve()}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the multi-command CLI parser."""
    parser = argparse.ArgumentParser(
        prog="talad",
        description="TALAD: This American Life Audio Dataset Downloader & Management Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. download subcommand
    dl_parser = subparsers.add_parser(
        "download",
        help="Download dataset splits or specific episodes and optionally transcode to WAV",
    )
    _add_download_arguments(dl_parser, require_target=False)
    dl_parser.set_defaults(func=_run_download)

    # 2. status / verify subcommands
    for cmd_name in ("status", "verify"):
        st_parser = subparsers.add_parser(
            cmd_name,
            help="Inspect downloaded dataset splits, completion rates, and missing episodes",
        )
        st_parser.add_argument(
            "-p",
            "--download_path",
            default="./data",
            help="Root dataset directory to inspect (default: ./data)",
        )
        st_parser.add_argument(
            "-d",
            "--splits",
            nargs="+",
            default=None,
            help="Splits to inspect (default: test valid train)",
        )
        st_parser.add_argument(
            "--show-missing",
            action="store_true",
            help="Print full list of missing and corrupted episode IDs",
        )
        st_parser.add_argument("-v", "--verbose", action="store_true")
        st_parser.set_defaults(func=_run_status)

    # 3. catalog subcommand
    cat_parser = subparsers.add_parser(
        "catalog",
        help="Search the episode catalog, inspect episode metadata, or update URLs",
    )
    cat_parser.add_argument("-q", "--query", default=None, help="Search keyword in title/description")
    cat_parser.add_argument("-e", "--episode", default=None, help="Inspect a single episode ID (e.g. 704)")
    cat_parser.add_argument(
        "-s",
        "--split",
        choices=["train", "valid", "test"],
        default=None,
        help="Filter results by dataset split",
    )
    cat_parser.add_argument("-n", "--limit", type=int, default=25, help="Max results to display (default: 25)")
    cat_parser.add_argument("--update", action="store_true", help="Update catalog CSV from upstream GitHub")
    cat_parser.add_argument("-v", "--verbose", action="store_true")
    cat_parser.set_defaults(func=_run_catalog)

    # 4. manifest subcommand
    man_parser = subparsers.add_parser(
        "manifest",
        help="Export JSONL/CSV/JSON audio manifests for speech recognition & diarization pipelines",
    )
    man_parser.add_argument(
        "-p",
        "--download_path",
        default="./data",
        help="Root dataset directory containing split folders (default: ./data)",
    )
    man_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output manifest file path (e.g. ./data/manifest.jsonl)",
    )
    man_parser.add_argument(
        "-f",
        "--format",
        choices=["jsonl", "json", "csv"],
        default="jsonl",
        help="Manifest output format (default: jsonl)",
    )
    man_parser.add_argument(
        "-d",
        "--splits",
        nargs="+",
        default=None,
        help="Splits to include (default: train valid test)",
    )
    man_parser.add_argument(
        "--probe-duration",
        action="store_true",
        help="Compute exact audio duration in seconds using wave/ffprobe",
    )
    man_parser.add_argument("-v", "--verbose", action="store_true")
    man_parser.set_defaults(func=_run_manifest)

    # 5. convert subcommand
    conv_parser = subparsers.add_parser(
        "convert",
        help="Batch-convert an existing folder of MP3 episodes to WAV/FLAC/OGG",
    )
    conv_parser.add_argument("folder", help="Path to directory containing .mp3 files")
    conv_parser.add_argument("-ar", "--sample-rate", type=int, default=16000, help="Sample rate (default: 16000)")
    conv_parser.add_argument("-ac", "--channels", type=int, default=1, help="Audio channels (default: 1)")
    conv_parser.add_argument("-ab", "--bit-depth", type=int, default=16, help="Bit depth (default: 16)")
    conv_parser.add_argument("--format", choices=["wav", "flac", "ogg"], default="wav")
    conv_parser.add_argument("--keep-mp3", action="store_true", help="Keep source MP3 files")
    conv_parser.add_argument("-j", "--workers", type=int, default=2, help="Parallel workers (default: 2)")
    conv_parser.add_argument("-v", "--verbose", action="store_true")
    conv_parser.set_defaults(func=_run_convert)

    # 6. clean subcommand
    clean_parser = subparsers.add_parser(
        "clean",
        help="Remove downloaded audio files for specified dataset splits",
    )
    clean_parser.add_argument("-p", "--download_path", required=True, help="Root dataset directory")
    clean_parser.add_argument(
        "-d",
        "--splits",
        nargs="+",
        required=True,
        help="Splits to remove (e.g. test valid)",
    )
    clean_parser.add_argument("-v", "--verbose", action="store_true")
    clean_parser.set_defaults(func=_run_clean)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint supporting both modern subcommands and legacy flag syntax."""
    args_list: List[str] = list(argv) if argv is not None else sys.argv[1:]

    # Backward compatibility: if invoked directly with flags (e.g., `-p ./dir -d test`),
    # route transparently to the `download` command parser.
    if args_list and args_list[0] not in SUBCOMMANDS and not args_list[0] in ("-h", "--help"):
        legacy_parser = argparse.ArgumentParser(
            prog="TAL_download_audio.py",
            description="This American Life Audio Dataset Downloader",
        )
        _add_download_arguments(legacy_parser, require_target=False)
        parsed = legacy_parser.parse_args(args_list)
        return _run_download(parsed)

    parser = build_parser()
    parsed = parser.parse_args(args_list)
    return int(parsed.func(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
