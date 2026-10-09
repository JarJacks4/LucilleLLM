"""
Move Escape's images and audio from the old project (escape-self-care-ai) to the app's project
(escape-self-care-505618), and re-point the links.

Run in Google Cloud Shell as someone who can READ the old project's Storage/Firestore and WRITE the new
project's (an Owner of both is simplest). Nothing is deleted from the old project.

    pip install --user google-cloud-storage google-cloud-firestore
    python scripts/gcp/migrate_media.py inventory              # what's there, sizes, Firestore collections
    python scripts/gcp/migrate_media.py copy                   # server-side copy, keeps download tokens
    python scripts/gcp/migrate_media.py verify                 # every object: size, md5, token match
    python scripts/gcp/migrate_media.py firestore              # dry run: links to rewrite in the app's Firestore
    python scripts/gcp/migrate_media.py firestore --apply      # rewrite them
    python scripts/gcp/migrate_media.py copy-collections Music SelfCareGoals   # optional: old Firestore -> new

Why links barely change: a Firebase download URL is
    https://firebasestorage.googleapis.com/v0/b/<BUCKET>/o/<path>?alt=media&token=<T>
and the token lives in the object's metadata (firebaseStorageDownloadTokens). The copy keeps that metadata,
so the new URL is the old one with only <BUCKET> swapped. `verify` proves it per object.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import quote

OLD_PROJECT = "escape-self-care-ai"
NEW_PROJECT = "escape-self-care-505618"
OLD_BUCKETS = ["escape-self-care-ai.firebasestorage.app", "escape-self-care-ai.appspot.com"]
NEW_BUCKET = "escape-self-care-505618.firebasestorage.app"
OUT = Path("media_migration")
TOKEN_KEY = "firebaseStorageDownloadTokens"


def storage():
    from google.cloud import storage as gcs
    return gcs


def existing_old_buckets(client):
    out = []
    for name in OLD_BUCKETS:
        try:
            client.get_bucket(name)
            out.append(name)
        except Exception as e:
            print(f"  (skipping gs://{name}: {type(e).__name__})")
    return out


def cmd_inventory(_):
    gcs = storage()
    src = gcs.Client(project=OLD_PROJECT)
    OUT.mkdir(exist_ok=True)
    total_n = total_b = 0
    with open(OUT / "inventory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "path", "bytes", "content_type", "has_token", "md5"])
        for b in existing_old_buckets(src):
            n = size = 0
            folders = {}
            for blob in src.list_blobs(b):
                n += 1
                size += blob.size or 0
                top = blob.name.split("/")[0] if "/" in blob.name else "(root)"
                folders[top] = folders.get(top, 0) + 1
                w.writerow([b, blob.name, blob.size, blob.content_type,
                            bool((blob.metadata or {}).get(TOKEN_KEY)), blob.md5_hash])
            total_n += n
            total_b += size
            print(f"gs://{b}: {n} objects, {size / 1e6:.1f} MB")
            for k, v in sorted(folders.items(), key=lambda kv: -kv[1])[:20]:
                print(f"    {v:5d}  {k}")
    print(f"TOTAL {total_n} objects, {total_b / 1e6:.1f} MB  -> {OUT}/inventory.csv")
    try:
        from google.cloud import firestore
        db = firestore.Client(project=OLD_PROJECT)
        cols = list(db.collections())
        print(f"\nOld project Firestore: {len(cols)} root collections")
        for c in cols:
            cnt = c.count().get()[0][0].value
            print(f"    {cnt:6d}  {c.id}")
    except Exception as e:
        print(f"\n(Old Firestore not readable: {type(e).__name__})")


def cmd_copy(args):
    gcs = storage()
    src = gcs.Client(project=OLD_PROJECT)
    dst = gcs.Client(project=NEW_PROJECT)
    dst_bucket = dst.bucket(NEW_BUCKET)
    copied = skipped = failed = 0
    for b in existing_old_buckets(src):
        sb = src.bucket(b)
        for blob in src.list_blobs(b):
            target = dst_bucket.get_blob(blob.name)
            if target is not None and target.md5_hash == blob.md5_hash and not args.force:
                skipped += 1
                continue
            try:
                # server-side rewrite: no download, metadata (incl. download token) carried over
                new = sb.copy_blob(blob, dst_bucket, blob.name)
                src_meta = blob.metadata or {}
                if src_meta.get(TOKEN_KEY) and (new.metadata or {}).get(TOKEN_KEY) != src_meta[TOKEN_KEY]:
                    new.metadata = {**(new.metadata or {}), **src_meta}
                    new.patch()
                copied += 1
                if copied % 50 == 0:
                    print(f"  copied {copied}…")
            except Exception as e:
                failed += 1
                print(f"  FAILED {b}/{blob.name}: {e}")
    print(f"copied {copied}, already there {skipped}, failed {failed}")
    sys.exit(1 if failed else 0)


def new_url(old_bucket: str, path: str, token: str | None) -> str:
    u = f"https://firebasestorage.googleapis.com/v0/b/{NEW_BUCKET}/o/{quote(path, safe='')}?alt=media"
    return u + (f"&token={token}" if token else "")


def cmd_verify(_):
    gcs = storage()
    src = gcs.Client(project=OLD_PROJECT)
    dst = gcs.Client(project=NEW_PROJECT).bucket(NEW_BUCKET)
    OUT.mkdir(exist_ok=True)
    bad = ok = 0
    with open(OUT / "url_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_bucket", "path", "token_match", "new_url"])
        for b in existing_old_buckets(src):
            for blob in src.list_blobs(b):
                t = dst.get_blob(blob.name)
                tok = (blob.metadata or {}).get(TOKEN_KEY)
                good = t is not None and t.size == blob.size and t.md5_hash == blob.md5_hash
                tmatch = (not tok) or ((t.metadata or {}).get(TOKEN_KEY) == tok if t else False)
                if good and tmatch:
                    ok += 1
                else:
                    bad += 1
                    print(f"  MISMATCH {blob.name}: exists={t is not None} same_bytes={good} token={tmatch}")
                w.writerow([b, blob.name, tmatch, new_url(b, blob.name, (tok or "").split(",")[0] or None)])
    print(f"verified {ok} objects, {bad} problems -> {OUT}/url_map.csv")
    sys.exit(1 if bad else 0)


def _rewrite(v):
    """Swap the bucket in any Firebase/GCS URL inside strings, lists and maps. Returns (value, changed)."""
    if isinstance(v, str):
        n = v
        for ob in OLD_BUCKETS:
            n = n.replace(f"/v0/b/{ob}/", f"/v0/b/{NEW_BUCKET}/").replace(f"storage.googleapis.com/{ob}/",
                                                                         f"storage.googleapis.com/{NEW_BUCKET}/")
            n = n.replace(f"gs://{ob}/", f"gs://{NEW_BUCKET}/")
        return n, n != v
    if isinstance(v, list):
        out, ch = [], False
        for x in v:
            y, c = _rewrite(x)
            out.append(y)
            ch |= c
        return out, ch
    if isinstance(v, dict):
        out, ch = {}, False
        for k, x in v.items():
            y, c = _rewrite(x)
            out[k] = y
            ch |= c
        return out, ch
    return v, False


def _walk(coll, fn):
    for doc in coll.stream():
        fn(doc)
        for sub in doc.reference.collections():
            _walk(sub, fn)


def cmd_firestore(args):
    from google.cloud import firestore
    db = firestore.Client(project=NEW_PROJECT)
    hits = []

    def visit(doc):
        data = doc.to_dict() or {}
        changes = {}
        for k, v in data.items():
            nv, ch = _rewrite(v)
            if ch:
                changes[k] = nv
        if changes:
            hits.append((doc.reference.path, list(changes)))
            if args.apply:
                doc.reference.update(changes)

    for c in db.collections():
        _walk(c, visit)
    OUT.mkdir(exist_ok=True)
    (OUT / "firestore_rewrites.json").write_text(json.dumps(hits, indent=1))
    print(f"{'rewrote' if args.apply else 'would rewrite'} {len(hits)} documents -> {OUT}/firestore_rewrites.json")
    if not args.apply and hits:
        print("Run again with --apply to write the changes.")


def cmd_copy_collections(args):
    from google.cloud import firestore
    src = firestore.Client(project=OLD_PROJECT)
    dst = firestore.Client(project=NEW_PROJECT)
    for name in args.collections:
        n = skipped = 0
        for doc in src.collection(name).stream():
            ref = dst.collection(name).document(doc.id)
            if ref.get().exists and not args.overwrite:
                skipped += 1
                continue
            data, _ = _rewrite(doc.to_dict() or {})
            ref.set(data)
            n += 1
        print(f"{name}: copied {n}, kept existing {skipped}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory")
    c = sub.add_parser("copy")
    c.add_argument("--force", action="store_true", help="re-copy even if already identical")
    sub.add_parser("verify")
    fs = sub.add_parser("firestore")
    fs.add_argument("--apply", action="store_true")
    cc = sub.add_parser("copy-collections")
    cc.add_argument("collections", nargs="+")
    cc.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    {"inventory": cmd_inventory, "copy": cmd_copy, "verify": cmd_verify, "firestore": cmd_firestore,
     "copy-collections": cmd_copy_collections}[a.cmd](a)


if __name__ == "__main__":
    main()
