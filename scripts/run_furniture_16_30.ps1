# Run OpenAI pipeline on furniture 16–30, skip 28 (already done).
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $env:OPENAI_API_KEY) {
    Write-Error "Set OPENAI_API_KEY first: `$env:OPENAI_API_KEY = 'sk-...'"
    exit 1
}

$outDir = "output\Exeprements_luna_56"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Copy existing product 28 result if present
$done28 = "output\28_furniture_pipeline.json"
if (Test-Path $done28) {
    Copy-Item $done28 "$outDir\" -Force
    Write-Host "Copied existing: $outDir\28_furniture_pipeline.json"
}

foreach ($n in 16..30) {
    if ($n -eq 28) {
        Write-Host "Skipping 28_furniture.json (already done)"
        continue
    }

    $product = "data\products\${n}_furniture.json"
    if (-not (Test-Path $product)) {
        Write-Error "Missing: $product"
        exit 1
    }

    Write-Host "`n========== Running $product =========="
    & .\.venv\Scripts\python.exe -m atsn.pipeline --backend openai_api --product $product
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Pipeline failed on $product"
        exit $LASTEXITCODE
    }

    Copy-Item "output\${n}_furniture_pipeline.json" "$outDir\" -Force
    Write-Host "Saved: $outDir\${n}_furniture_pipeline.json"
}

Write-Host "`nDone. Furniture 16–30 (28 skipped) in $outDir"
