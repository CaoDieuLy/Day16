param(
    [string]$ProjectId = "ai-lab-16-gcp-494213"
)

$ErrorActionPreference = "Stop"

$terraform = "terraform"
$instanceName = & $terraform output -raw gpu_node_name
$zone = & $terraform output -raw gpu_node_zone

Write-Host "Running benchmark on instance $instanceName in zone $zone..."
gcloud compute ssh $instanceName --zone=$zone --project=$ProjectId --tunnel-through-iap --command "cd /opt/ml-benchmark && sudo ./run_benchmark.sh"

Write-Host "Fetching benchmark_result.json..."
$result = gcloud compute ssh $instanceName --zone=$zone --project=$ProjectId --tunnel-through-iap --command "sudo cat /opt/ml-benchmark/benchmark_result.json"
$result | Set-Content -Path "benchmark_result.json"

Write-Host "Fetching benchmark_report.txt..."
$report = gcloud compute ssh $instanceName --zone=$zone --project=$ProjectId --tunnel-through-iap --command "sudo cat /opt/ml-benchmark/benchmark_report.txt"
$report | Set-Content -Path "benchmark_report.txt"

Write-Host "Artifacts saved to:"
Write-Host " - $(Join-Path (Get-Location) 'benchmark_result.json')"
Write-Host " - $(Join-Path (Get-Location) 'benchmark_report.txt')"
