$ErrorActionPreference = "Stop"

$modelDir = Join-Path $PSScriptRoot "models/multilingual-e5-small"
New-Item -ItemType Directory -Path $modelDir -Force | Out-Null

$base = "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx"
$files = @(
    "model_O4.onnx",
    "tokenizer.json"
)

foreach ($file in $files) {
    $target = Join-Path $modelDir $file
    if (Test-Path $target) {
        Write-Host "$file already exists"
        continue
    }

    Write-Host "Downloading $file ..."
    Invoke-WebRequest -Uri "$base/$file?download=true" -OutFile $target
}

Write-Host "Done. Model is in $modelDir"
