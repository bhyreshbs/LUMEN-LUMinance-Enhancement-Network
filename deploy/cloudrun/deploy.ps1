<#
.SYNOPSIS
  Build the LUMEN image with Cloud Build, push it to Artifact Registry and deploy it to Cloud Run.

.DESCRIPTION
  Uses the repository's production Dockerfile unchanged. The checkpoint comes from the private bucket
  created by setup.ps1, mounted read-only at /models (Cloud Storage volume, gen2 execution environment),
  where the app verifies its SHA-256 at start-up. Without it, only the Quick (classical) style is offered.

  Resource settings (validated against the measured container profile in docs/DEPLOYMENT.md - peak 1.7-2.6 GB
  at 768 px with 2 CPUs, so 4 GiB leaves >= 1.4 GiB headroom):
    2 vCPU, 4 GiB, concurrency 1, min 0 / max 2 instances, 300 s timeout, startup CPU boost, x86_64 Linux.

  -Public makes the service reachable without sign-in (needed for a public demo URL). Without it the service
  requires IAM authentication.

.EXAMPLE
  .\deploy\cloudrun\deploy.ps1 -ProjectId my-project-id -Public
#>
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [string]$Region = "asia-south1",
    [string]$Bucket = "",
    [string]$Service = "lumen",
    [string]$Tag = "",
    [int]$Concurrency = 1,
    [int]$MaxInstances = 2,
    [switch]$Public
)
$ErrorActionPreference = "Stop"
if (-not $Bucket) { $Bucket = "$ProjectId-lumen-weights" }
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not $Tag) {
    $Tag = (& git -C $Root rev-parse --short HEAD 2>$null)
    if (-not $Tag) { $Tag = Get-Date -Format "yyyyMMdd-HHmmss" }
}
$Image = "$Region-docker.pkg.dev/$ProjectId/lumen/lumen:$Tag"
$RunSa = "lumen-run@$ProjectId.iam.gserviceaccount.com"

function G {
    & gcloud @args
    if ($LASTEXITCODE -ne 0) { throw "gcloud $($args -join ' ') failed (exit $LASTEXITCODE)" }
}

Write-Host "== Building $Image with Cloud Build (uploads the repo minus .gcloudignore)" -ForegroundColor Cyan
Push-Location $Root
try { G builds submit --project $ProjectId --tag $Image . } finally { Pop-Location }

Write-Host "== Deploying Cloud Run service '$Service' in $Region" -ForegroundColor Cyan
$auth = if ($Public) { "--allow-unauthenticated" } else { "--no-allow-unauthenticated" }
G run deploy $Service --project $ProjectId --region $Region --image $Image `
    --execution-environment gen2 --cpu 2 --memory 4Gi --cpu-boost `
    --concurrency $Concurrency --min-instances 0 --max-instances $MaxInstances --timeout 300 --port 8080 `
    --service-account $RunSa `
    --set-env-vars "LUMEN_TORCH_THREADS=2,LUMEN_MAX_SIDE=768,LUMEN_RETINEXFORMER_WEIGHTS=/models/retinexformer_LOL_v1.pth" `
    --add-volume "name=models,type=cloud-storage,bucket=$Bucket,readonly=true" `
    --add-volume-mount "volume=models,mount-path=/models" `
    $auth

$Url = (& gcloud run services describe $Service --project $ProjectId --region $Region --format "value(status.url)").Trim()
if ($Public) {
    Write-Host "== Checking $Url" -ForegroundColor Cyan
    # /healthz is reserved by Google's frontend on *.run.app (404), so readiness is checked with /api/status
    $status = Invoke-RestMethod "$Url/api/status" -TimeoutSec 120    # first request may include a cold start
    $styles = ($status.styles | ForEach-Object { "$($_.name)=$(if ($_.available) {'available'} else {'UNAVAILABLE'})" }) -join ", "
    Write-Host "styles:  $styles"
} else {
    Write-Host "Service requires authentication. To test it locally:  gcloud run services proxy $Service --region $Region --port 7860"
}
Write-Host ""
Write-Host "Deployed: $Url" -ForegroundColor Green
Write-Host "Full check:  python scripts\smoke_test_app.py --url $Url --expect-modes standard quality_guided classical"
