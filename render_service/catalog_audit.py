"""Read-only GCS inventory check for the 159-track Lucille Soundscapes catalog.

The catalog is deployed with the API, but its M4A files are uploaded separately.

    python -m render_service.catalog_audit --bucket escape-self-care-505618-escape-media
    python -m render_service.catalog_audit --bucket escape-self-care-505618-escape-media --commands
    python -m render_service.catalog_audit --bucket escape-self-care-505618-escape-media --show-paths

The --commands option *prints* (never runs) batches of explicit render/upload
commands for tracks with at least one missing asset. Rendering takes CPU/time
and incurs GCP storage costs; review commands before executing them.
Requires Google Application Default Credentials with storage.objects.list.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Iterable

CATALOG = Path(__file__).resolve().parents[1] / "escape_api" / "data" / "soundscape_catalog.json"


def object_paths(track: dict) -> list[str]:
    """Return all seven actual playback assets for one catalog track."""
    audio = track["audio"]
    segments = audio["segments"]
    names = [segments["intro"], *segments["bodies"], segments["outro"], audio["preview"]]
    prefix = audio["path"].strip("/")
    return [f"{prefix}/{name}" for name in names]


def missing_by_track(tracks: list[dict], existing: Iterable[str]) -> dict[str, list[str]]:
    """Return only tracks with missing GCS objects; pure and easily tested."""
    existing_set = set(existing)
    return {
        track["id"]: missing
        for track in tracks
        if (missing := [name for name in object_paths(track) if name not in existing_set])
    }


def render_commands(track_ids: list[str], bucket: str, batch_size: int = 12) -> list[str]:
    """Explicit, opt-in renderer CLI commands grouped to keep them readable."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    prefix = f"AUDIO_BUCKET={shlex.quote(bucket)} python -m render_service.catalog --only"
    return [
        prefix + " " + " ".join(shlex.quote(track_id) for track_id in track_ids[i:i + batch_size])
        for i in range(0, len(track_ids), batch_size)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Soundscapes audio assets in GCS (read-only).")
    parser.add_argument("--bucket", default=os.getenv("AUDIO_BUCKET"),
                        help="GCS bucket containing soundscapes/ assets")
    parser.add_argument("--commands", action="store_true",
                        help="print (do not execute) commands to render/upload incomplete tracks")
    parser.add_argument("--show-paths", action="store_true", help="list each missing object path")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--fail-if-missing", action="store_true", help="exit 1 on any missing objects")
    args = parser.parse_args()

    if not args.bucket:
        parser.error("Specify --bucket or the AUDIO_BUCKET environment variable.")
    if args.batch_size < 1:
        parser.error("--batch-size must be greater than zero.")

    tracks = json.loads(CATALOG.read_text(encoding="utf-8"))["tracks"]
    wanted = [path for track in tracks for path in object_paths(track)]
    if len(wanted) != len(set(wanted)):
        raise RuntimeError("Catalog contains duplicate GCS asset paths.")
    try:
        from google.cloud import storage
        client = storage.Client()
        available = {blob.name for blob in client.list_blobs(args.bucket, prefix="soundscapes/")}
    except Exception as exc:
        print(f"Could not list audio objects in {args.bucket!r}: {type(exc).__name__}: {exc}")
        return 2

    absent = missing_by_track(tracks, available)
    missing_count = sum(map(len, absent.values()))
    print(f"Bucket: gs://{args.bucket}")
    print(f"Catalog: {len(tracks)} tracks, {len(wanted)} expected playback assets")
    print(f"Found: {len(wanted) - missing_count} expected assets")
    print(f"Missing: {missing_count} assets across {len(absent)} tracks")
    for track_id, names in absent.items():
        print(f"  {track_id}: {len(names)}/{len(object_paths(next(t for t in tracks if t['id'] == track_id)))} missing")
        if args.show_paths:
            for name in names:
                print(f"    gs://{args.bucket}/{name}")

    if args.commands and absent:
        print("\nReview before running: commands below generate and UPLOAD all seven files")
        print("for each incomplete track; they can be CPU-intensive and incur costs.")
        for command in render_commands(list(absent), args.bucket, args.batch_size):
            print(command)
    print("Note: GCS object presence does not guarantee public HTTP playback access.")
    return 1 if args.fail_if_missing and absent else 0


if __name__ == "__main__":
    raise SystemExit(main())
