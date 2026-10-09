"""
Re-point every image/audio link in the FlutterFlow project from the old bucket to the new one,
through FlutterFlow's Project API (paid plans; editor access). Dry run by default.

    export FF_API_TOKEN=...            # FlutterFlow > Account > API token
    export FF_PROJECT_ID=...           # FlutterFlow > Settings > Project ID
    python scripts/flutterflow_swap_media_urls.py            # dry run: which files, how many links
    python scripts/flutterflow_swap_media_urls.py --apply    # validate each file, then update

Run AFTER `migrate_media.py copy` and `verify` pass: the new links only work once the files are copied.
Only the bucket name changes; paths and download tokens stay the same. Libraries used by the app
(e.g. that_audio_player) are separate FlutterFlow projects: run again with their project id.
"""

import argparse
import base64
import io
import json
import os
import sys
import zipfile

import requests

API = "https://api.flutterflow.io/v2"
SWAPS = [
    ("/v0/b/escape-self-care-ai.firebasestorage.app/", "/v0/b/escape-self-care-505618.firebasestorage.app/"),
    ("/v0/b/escape-self-care-ai.appspot.com/", "/v0/b/escape-self-care-505618.firebasestorage.app/"),
    ("lucillellm2-286076426888.us-east4.run.app", "lucille-861854898360.us-central1.run.app"),
]


def call(method, path, token, **kw):
    r = requests.request(method, f"{API}{path}", headers={"Authorization": f"Bearer {token}"}, timeout=120, **kw)
    if r.status_code >= 400:
        sys.exit(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--project", default=os.getenv("FF_PROJECT_ID"))
    a = ap.parse_args()
    token = os.getenv("FF_API_TOKEN")
    if not token or not a.project:
        sys.exit("Set FF_API_TOKEN and FF_PROJECT_ID (or --project).")
    data = call("GET", "/projectYamls", token, params={"projectId": a.project})
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(data["value"]["projectYamlBytes"])))
    changed = {}
    total = 0
    for name in z.namelist():
        if not name.endswith((".yaml", ".yml")):
            continue
        text = z.read(name).decode("utf-8")
        new, n = text, 0
        for old, rep in SWAPS:
            n += new.count(old)
            new = new.replace(old, rep)
        if n:
            key = name.rsplit(".", 1)[0]
            changed[key] = new
            total += n
            print(f"{n:4d}  {key}")
    print(f"{total} links in {len(changed)} files")
    os.makedirs("media_migration", exist_ok=True)
    with open("media_migration/flutterflow_changes.json", "w") as f:
        json.dump({k: v for k, v in changed.items()}, f)
    if not a.apply or not changed:
        print("Dry run. Re-run with --apply to update the FlutterFlow project.")
        return
    for key, content in changed.items():
        v = call("POST", "/validateProjectYaml", token, json={"projectId": a.project, "fileKey": key, "fileContent": content})
        if not v.get("success"):
            sys.exit(f"Validation failed for {key}: {json.dumps(v)[:800]}")
    keys = list(changed)
    for i in range(0, len(keys), 20):        # small batches
        batch = {k: changed[k] for k in keys[i:i + 20]}
        call("POST", "/updateProjectByYaml", token, json={"projectId": a.project, "fileKeyToContent": batch})
        print(f"updated {min(i + 20, len(keys))}/{len(keys)} files")
    print("Done. Open FlutterFlow, check a few images and tracks, then push to GitHub from FlutterFlow.")


if __name__ == "__main__":
    main()
