#!/usr/bin/env bash
# One-time setup so GitHub Actions can deploy LucilleLLM to Cloud Run WITHOUT a JSON key.
# Run in Google Cloud Shell (https://shell.cloud.google.com) as a project Owner:
#
#   bash scripts/gcp/setup_deploy_access.sh            # project: escape-self-care-505618
#
# It creates:
#   * secrets 'lucille-render-api-key' + 'lucille-render-callback-secret' (API <-> render service)
#   * public bucket <PROJECT_ID>-escape-media for orbs, loops and audio
#   * service account  github-deployer@<PROJECT_ID>.iam.gserviceaccount.com  (least privilege)
#   * Workload Identity pool 'github' + provider 'github-oidc', locked to repo JarJacks4/LucilleLLM
#   * secret 'internal-service-key' (server-to-server auth for scheduled jobs)
#   * Cloud Scheduler job 'lucille-letters-due' (hourly) -> POST /v1/jobs/letters-due
# and prints the three values to paste into GitHub -> Settings -> Secrets and variables -> Actions.
set -euo pipefail

PROJECT_ID="${1:-escape-self-care-505618}"
if [ "$PROJECT_ID" != "escape-self-care-505618" ] && [ "${ALLOW_OTHER_PROJECT:-}" != "1" ]; then
  echo "This sets up deploys for escape-self-care-505618 (the app's Firebase project). Got $PROJECT_ID; set ALLOW_OTHER_PROJECT=1 to override."; exit 1
fi
REPO="${REPO:-JarJacks4/LucilleLLM}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-lucille}"
SA_NAME="github-deployer"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
POOL="github"
PROVIDER="github-oidc"

gcloud config set project "$PROJECT_ID" >/dev/null
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
echo "Project $PROJECT_ID ($PROJECT_NUMBER)"

echo "== Enabling APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com iamcredentials.googleapis.com sts.googleapis.com cloudscheduler.googleapis.com

echo "== Artifact Registry repo (lucille-repo) if missing"
gcloud artifacts repositories describe lucille-repo --location="$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create lucille-repo --repository-format=docker --location="$REGION"

echo "== Deployer service account"
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name="GitHub Actions deployer (LucilleLLM)"
for ROLE in roles/cloudbuild.builds.editor roles/run.admin roles/artifactregistry.writer \
            roles/storage.admin roles/logging.viewer roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$SA" --role="$ROLE" --condition=None >/dev/null
done
# Cloud Build deploys as the Cloud Build SA / compute SA; the deployer must be able to act as them.
for RUNTIME in "${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" "${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"; do
  gcloud iam service-accounts add-iam-policy-binding "$RUNTIME" --member="serviceAccount:$SA" \
    --role=roles/iam.serviceAccountUser >/dev/null 2>&1 || true
done
# Cloud Build's own SA needs to deploy to Cloud Run and read secrets
CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
for ROLE in roles/run.admin roles/iam.serviceAccountUser roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$CB_SA" --role="$ROLE" --condition=None >/dev/null
done
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor --condition=None >/dev/null

echo "== Workload Identity Federation (keyless GitHub -> GCP)"
gcloud iam workload-identity-pools describe "$POOL" --location=global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools create "$POOL" --location=global --display-name="GitHub"
gcloud iam workload-identity-pools providers describe "$PROVIDER" --location=global --workload-identity-pool="$POOL" >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location=global \
    --workload-identity-pool="$POOL" --display-name="GitHub OIDC" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition="assertion.repository == '${REPO}'"
gcloud iam service-accounts add-iam-policy-binding "$SA" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${REPO}" >/dev/null

echo "== Secrets"
gcloud secrets describe openai-api-key >/dev/null 2>&1 || echo "!! secret 'openai-api-key' is missing. Create it: echo -n sk-... | gcloud secrets create openai-api-key --data-file=-"
if ! gcloud secrets describe internal-service-key >/dev/null 2>&1; then
  head -c 32 /dev/urandom | base64 | tr -d '\n=/+' | gcloud secrets create internal-service-key --data-file=- --replication-policy=automatic
fi

for S in lucille-render-api-key lucille-render-callback-secret; do
  gcloud secrets describe "$S" >/dev/null 2>&1 || \
    head -c 32 /dev/urandom | base64 | tr -d '\n=/+' | gcloud secrets create "$S" --data-file=- --replication-policy=automatic
done

echo "== Public media bucket for orbs, loops and soundscape audio (kept apart from the Firebase bucket)"
MEDIA="gs://${PROJECT_ID}-escape-media"
gcloud storage buckets describe "$MEDIA" >/dev/null 2>&1 || \
  gcloud storage buckets create "$MEDIA" --location="$REGION" --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding "$MEDIA" --member=allUsers --role=roles/storage.objectViewer >/dev/null
gcloud storage buckets add-iam-policy-binding "$MEDIA" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" --role=roles/storage.objectAdmin >/dev/null
echo "   media base URL: https://storage.googleapis.com/${PROJECT_ID}-escape-media"

echo "== Cloud Scheduler: hourly letters-to-future-self delivery"
URL="$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)' 2>/dev/null || true)"
if [ -n "$URL" ]; then
  KEY="$(gcloud secrets versions access latest --secret=internal-service-key)"
  gcloud scheduler jobs describe lucille-letters-due --location="$REGION" >/dev/null 2>&1 && \
    gcloud scheduler jobs delete lucille-letters-due --location="$REGION" --quiet
  gcloud scheduler jobs create http lucille-letters-due --location="$REGION" --schedule="7 * * * *" \
    --uri="${URL}/v1/jobs/letters-due" --http-method=POST --message-body='{"dryRun":false}' \
    --headers="Content-Type=application/json,Authorization=Bearer ${KEY}" >/dev/null
  echo "   scheduled -> ${URL}/v1/jobs/letters-due"
else
  echo "   (service '$SERVICE' not found yet; re-run this script after the first deploy to add the job)"
fi

cat <<EOF

================  Paste these into GitHub  ================
Repo: https://github.com/${REPO}/settings/secrets/actions  ->  New repository secret

  GCP_WIF_PROVIDER      projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}
  GCP_DEPLOY_SA         ${SA}

Then: Actions -> "Deploy to Cloud Run" -> Run workflow (or merge to main).
No JSON keys were created; GitHub gets short-lived tokens only for ${REPO}.
===========================================================
EOF
