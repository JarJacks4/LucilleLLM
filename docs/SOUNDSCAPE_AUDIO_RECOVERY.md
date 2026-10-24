# Soundscapes audio recovery (production)

The production API returns the complete `soundscape_catalog.json` (159 tracks,
11 categories), but **the catalog does not contain audio bytes**. Each track
needs seven separately generated objects in Google Cloud Storage:

`soundscapes/<category>/<track_id>/{intro,body_1,body_2,body_3,body_4,outro,preview}.m4a`

The existing Cloud Run/CD pipeline deploys the API and the render service;
it does **not** pre-render the public catalog. A passing `/v1/soundscapes/categories`
health check therefore does not guarantee the tracks are playable.

## Audit before changing anything

From the LucilleLLM repo root, in an environment with Python 3.12,
`ffmpeg`, and Google Cloud Application Default Credentials:

```bash
python -m pip install -r render_service/requirements.txt
gcloud config set project escape-self-care-505618
gcloud auth application-default login
python -m render_service.catalog_audit --bucket escape-self-care-505618-escape-media --show-paths
```

The audit lists GCS objects under `soundscapes/` and compares them with
the seven paths defined for each track. It is **read-only** and uses
`storage.objects.list`. It distinguishes missing files from app catalog bugs,
but a present object may still fail over HTTP due to access permissions.

To list the exact batches of render commands without executing them:

```bash
python -m render_service.catalog_audit --bucket escape-self-care-505618-escape-media --commands
```

## Render and upload only affected tracks (manual approval required)

Review the emitted commands and run them individually from a machine with
`ffmpeg`, ample disk/CPU time, and credentials authorized to write objects to
the target bucket. Each command resembles:

```bash
AUDIO_BUCKET=escape-self-care-505618-escape-media python -m render_service.catalog --only jazz_01
```

The renderer writes the complete seven-file asset set to the catalog-defined
prefix, using AAC-LC in M4A. **Rendering is expensive:** six long segments plus
a 30-second preview per track; do not run the full catalog command blindly.
The helper emits only IDs with at least one missing object, although a
partially uploaded track is intentionally regenerated as a complete set.

If credentials or Storage permissions are missing, ask the authorized
Google Cloud administrator to run the commands. Do not add service-account
keys to the repository or publish bucket access tokens.

## Verify recovery

Run the same audit after rendering; it should report zero missing assets.
Spot-check several formerly missing `intro.m4a` and `preview.m4a` files
using their published `https://storage.googleapis.com/.../soundscapes/...`
URLs, then play them on both iOS and Android. A 403/404 on a **present** GCS
object indicates a public-delivery/IAM or URL issue, not missing rendering.

Voice synthesis is separate. Test `POST /v1/voice/speak` with appropriate
authentication/consent and then `POST /v1/voice/converse` from a device:
confirm that `audio` is non-empty, `audioMime` is `audio/mpeg`, and
the app actually plays the returned base64 MP3.
