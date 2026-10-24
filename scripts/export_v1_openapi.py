"""
Write docs/escape_v1_openapi.json: only the /v1 endpoints, ready for
FlutterFlow -> API Calls -> + Add -> Import OpenAPI / Swagger, and for Swift/Dart codegen.

    python scripts/export_v1_openapi.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  # noqa: E402

from escape_api import register  # noqa: E402
from escape_api.routers.meta import API_VERSION  # noqa: E402

BASE = os.getenv("PUBLIC_BASE_URL", "https://lucille-861854898360.us-central1.run.app")

app = FastAPI(title="Lucille · Escape v1 API", version=API_VERSION,
              description="Journal, Mood Scan + Mood Stats, Soundscapes AI, Self-Care Score + Plans, GDPR. "
                          "Send 'Authorization: Bearer <Firebase ID token>' and 'X-Timezone: <IANA tz>' on every call.")
register(app)
spec = app.openapi()
spec["servers"] = [{"url": BASE, "description": "Production (Cloud Run)"}]
spec["paths"] = {p: v for p, v in spec["paths"].items() if p.startswith("/v1")}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "escape_v1_openapi.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(spec, open(out, "w"), indent=1)
print(f"wrote {out} with {len(spec['paths'])} paths")
