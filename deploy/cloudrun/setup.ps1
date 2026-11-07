<#
.SYNOPSIS
  One-time Google Cloud setup for LUMEN on Cloud Run. Idempotent: existing resources are left as they are.

.DESCRIPTION
  Creates (only if missing), in the project you pass:
    * APIs: Cloud Run, Cloud Build, Artifact Registry, Cloud Storage, IAM, IAM Credentials, STS
    * Artifact Registry Docker repository "lumen" in -Region
    * a PRIVATE Cloud Storage bucket holding the Retinexformer checkpoint (uploaded from weights\ after a
      local SHA-256 check). The checkpoint is never put in the image or the repository (licence unclear).
    * service account lumen-run  (runtime identity: read-only access to that bucket, nothing else)
    * unless -SkipGitHub: service account lumen-deployer and a Workload Identity Federation pool/provider
      that only the GitHub repository -GitHubRepo can use (no long-lived keys anywhere)

  Billable resources it creates: the bucket (a 6.5 MB object - fractions of a cent per month) and the
  Artifact Registry repository (charged for stored images once you push them). It does NOT deploy anything.

.EXAMPLE
  .\deploy\cloudrun\setup.ps1 -ProjectId my-project-id
#>
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [string]$Region = "asia-south1",
    [string]$Bucket = "",
    [string]$GitHubRepo = "bhyreshbs/LUMEN-LUMinance-Enhancement-Network",
    [switch]$SkipGitHub
)
$ErrorActionPreference = "Stop"
if (-not $Bucket) { $Bucket = "$ProjectId-lumen-weights" }
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Checkpoint = Join-Path $Root "weights\retinexformer_LOL_v1.pth"
$ExpectedSha = "9b4810dc51322fab70be8658ee0ad1df64a54aa814497b7849767baa2ff32cd1"   # lumen/deep/checkpoints.py

function G {  # run gcloud, stop on failure
    & gcloud @args
    if ($LASTEXITCODE -ne 0) { throw "gcloud $($args -join ' ') failed (exit $LASTEXITCODE)" }
}
function Exists {  # true if the gcloud describe command succeeds
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"   # Windows PowerShell 5.1 turns redirected native stderr into errors
    & gcloud @args 2>$null | Out-Null
    $ok = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prev
    return $ok
}

Write-Host "== Project $ProjectId, region $Region" -ForegroundColor Cyan
G config set project $ProjectId
$billing = & gcloud billing projects describe $ProjectId --format="value(billingEnabled)"
if ($billing -ne "True") {
    throw "Billing is not enabled for $ProjectId. Link a billing account first: https://console.cloud.google.com/billing/linkedaccount?project=$ProjectId"
}
$ProjectNumber = (& gcloud projects describe $ProjectId --format="value(projectNumber)").Trim()

Write-Host "== Enabling APIs" -ForegroundColor Cyan
G services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com `
    storage.googleapis.com iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com

Write-Host "== Artifact Registry repository 'lumen'" -ForegroundColor Cyan
if (-not (Exists artifacts repositories describe lumen --location $Region)) {
    G artifacts repositories create lumen --repository-format docker --location $Region `
        --description "LUMEN container images"
}

Write-Host "== Private checkpoint bucket gs://$Bucket" -ForegroundColor Cyan
if (-not (Test-Path $Checkpoint)) { throw "Missing $Checkpoint - run: python scripts\download_weights.py --runtime" }
$sha = (Get-FileHash $Checkpoint -Algorithm SHA256).Hash.ToLower()
if ($sha -ne $ExpectedSha) { throw "SHA-256 mismatch for $Checkpoint (got $sha) - re-download it" }
if (-not (Exists storage buckets describe "gs://$Bucket")) {
    G storage buckets create "gs://$Bucket" --location $Region --uniform-bucket-level-access --public-access-prevention
}
G storage cp $Checkpoint "gs://$Bucket/retinexformer_LOL_v1.pth"

Write-Host "== Runtime service account lumen-run (read-only on the bucket)" -ForegroundColor Cyan
$RunSa = "lumen-run@$ProjectId.iam.gserviceaccount.com"
if (-not (Exists iam service-accounts describe $RunSa)) {
    G iam service-accounts create lumen-run --display-name "LUMEN Cloud Run runtime"
}
G storage buckets add-iam-policy-binding "gs://$Bucket" --member "serviceAccount:$RunSa" `
    --role roles/storage.objectViewer --quiet

if (-not $SkipGitHub) {
    Write-Host "== GitHub Actions deployer (Workload Identity Federation, repo $GitHubRepo only)" -ForegroundColor Cyan
    $DeploySa = "lumen-deployer@$ProjectId.iam.gserviceaccount.com"
    if (-not (Exists iam service-accounts describe $DeploySa)) {
        G iam service-accounts create lumen-deployer --display-name "LUMEN GitHub deployer"
    }
    # least privilege: push images to this one repository, deploy Cloud Run revisions, act as lumen-run only
    G artifacts repositories add-iam-policy-binding lumen --location $Region `
        --member "serviceAccount:$DeploySa" --role roles/artifactregistry.writer --quiet
    G projects add-iam-policy-binding $ProjectId --member "serviceAccount:$DeploySa" `
        --role roles/run.developer --condition=None --quiet | Out-Null
    G iam service-accounts add-iam-policy-binding $RunSa --member "serviceAccount:$DeploySa" `
        --role roles/iam.serviceAccountUser --quiet | Out-Null

    if (-not (Exists iam workload-identity-pools describe github --location global)) {
        G iam workload-identity-pools create github --location global --display-name "GitHub Actions"
    }
    if (-not (Exists iam workload-identity-pools providers describe github-oidc --location global --workload-identity-pool github)) {
        G iam workload-identity-pools providers create-oidc github-oidc --location global `
            --workload-identity-pool github --issuer-uri "https://token.actions.githubusercontent.com" `
            --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" `
            --attribute-condition "assertion.repository == '$GitHubRepo'"
    }
    $Principal = "principalSet://iam.googleapis.com/projects/$ProjectNumber/locations/global/workloadIdentityPools/github/attribute.repository/$GitHubRepo"
    G iam service-accounts add-iam-policy-binding $DeploySa --member $Principal `
        --role roles/iam.workloadIdentityUser --quiet | Out-Null
    $Provider = "projects/$ProjectNumber/locations/global/workloadIdentityPools/github/providers/github-oidc"
}

Write-Host ""
Write-Host "Setup complete. Nothing has been deployed yet." -ForegroundColor Green
Write-Host "Manual deploy:  .\deploy\cloudrun\deploy.ps1 -ProjectId $ProjectId -Public"
if (-not $SkipGitHub) {
    Write-Host ""
    Write-Host "For GitHub Actions, add these as repository *variables* (Settings > Secrets and variables > Actions > Variables):"
    Write-Host "  GCP_PROJECT_ID     = $ProjectId"
    Write-Host "  GCP_REGION         = $Region"
    Write-Host "  GCP_WIF_PROVIDER   = $Provider"
    Write-Host "  GCP_DEPLOY_SA      = $DeploySa"
    Write-Host "  GCP_RUN_SA         = $RunSa"
    Write-Host "  GCP_WEIGHTS_BUCKET = $Bucket"
    Write-Host "(These are identifiers, not secrets. No keys are created.)"
}
