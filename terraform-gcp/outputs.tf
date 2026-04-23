output "load_balancer_ip" {
  description = "External IP address of the Load Balancer (API endpoint)"
  value       = google_compute_global_forwarding_rule.vllm_fwd.ip_address
}

output "alb_ip_address" {
  description = "Alias kept for the README examples"
  value       = google_compute_global_forwarding_rule.vllm_fwd.ip_address
}

output "api_endpoint" {
  description = "vLLM API endpoint URL"
  value       = "http://${google_compute_global_forwarding_rule.vllm_fwd.ip_address}/v1"
}

output "endpoint_url" {
  description = "Chat completions endpoint URL"
  value       = "http://${google_compute_global_forwarding_rule.vllm_fwd.ip_address}/v1/chat/completions"
}

output "cpu_predict_url" {
  description = "CPU fallback LightGBM prediction endpoint URL"
  value       = "http://${google_compute_global_forwarding_rule.vllm_fwd.ip_address}/predict"
}

output "cpu_metrics_url" {
  description = "CPU fallback LightGBM benchmark metrics endpoint URL"
  value       = "http://${google_compute_global_forwarding_rule.vllm_fwd.ip_address}/metrics"
}

output "gpu_node_name" {
  description = "Name of the GPU Compute Engine instance"
  value       = google_compute_instance.gpu_node.name
}

output "gpu_node_zone" {
  description = "Zone of the GPU instance"
  value       = google_compute_instance.gpu_node.zone
}

output "iap_ssh_command" {
  description = "Command to SSH into the GPU node via IAP"
  value       = "gcloud compute ssh ${google_compute_instance.gpu_node.name} --zone=${google_compute_instance.gpu_node.zone} --project=${var.project_id} --tunnel-through-iap"
}

output "cpu_benchmark_run_command" {
  description = "Run the CPU fallback benchmark on the instance"
  value       = "gcloud compute ssh ${google_compute_instance.gpu_node.name} --zone=${google_compute_instance.gpu_node.zone} --project=${var.project_id} --tunnel-through-iap --command \"cd /opt/ml-benchmark && sudo ./run_benchmark.sh\""
}

output "cpu_benchmark_result_path" {
  description = "Path to the generated benchmark_result.json file on the instance"
  value       = "/opt/ml-benchmark/benchmark_result.json"
}

output "cpu_benchmark_report_path" {
  description = "Path to the generated short report on the instance"
  value       = "/opt/ml-benchmark/benchmark_report.txt"
}
