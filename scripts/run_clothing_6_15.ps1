# Run OpenAI pipeline on clothing products 6–15 and save to Exeprements_luna_56.
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $env:OPENAI_API_KEY) {
    Write-Error "Set OPENAI_API_KEY first: `$env:OPENAI_API_KEY = 'sk-...'"
    exit 1
}

$outDir = "output\Exeprements_luna_56"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

foreach ($n in 6..15) {
    $product = "data\products\${n}_clothing.json"
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

    Copy-Item "output\${n}_clothing_pipeline.json" "$outDir\" -Force
    Write-Host "Saved: $outDir\${n}_clothing_pipeline.json"
}

Write-Host "`nDone. 10 products (6–15) saved to $outDir"
