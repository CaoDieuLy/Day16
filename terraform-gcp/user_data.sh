#!/bin/bash
set -e
exec > >(tee /var/log/startup-script.log|logger -t startup-script -s 2>/dev/console) 2>&1

GPU_COUNT="${gpu_count}"

echo "Starting AI lab setup with gpu_count=$GPU_COUNT"

apt-get update -y

if [ "$GPU_COUNT" -gt 0 ]; then
  # Install Docker
  apt-get install -y docker.io
  systemctl enable docker
  systemctl start docker

  # Install NVIDIA container toolkit
  apt-get install -y linux-headers-$(uname -r)
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker

  # Run vLLM with the Gemma model
  docker run -d \
    --name vllm \
    --gpus all \
    --restart unless-stopped \
    -p 8000:8000 \
    --ipc=host \
    -e HF_TOKEN="${hf_token}" \
    -e HUGGING_FACE_HUB_TOKEN="${hf_token}" \
    vllm/vllm-openai:latest \
    --model "${model_id}" \
    --dtype half \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 8000

  echo "vLLM container started for model ${model_id}"
else
  # CPU fallback path: install a runnable benchmark plus a small demo inference API.
  apt-get install -y python3 python3-pip python3-venv unzip
  mkdir -p /opt/ml-benchmark
  python3 -m venv /opt/ml-benchmark/venv
  /opt/ml-benchmark/venv/bin/pip install --upgrade pip
  /opt/ml-benchmark/venv/bin/pip install fastapi uvicorn lightgbm scikit-learn pandas numpy

  cat >/opt/ml-benchmark/benchmark.py <<'PY'
${benchmark_py}
PY

  cat >/opt/ml-benchmark/run_benchmark.sh <<'RUN'
#!/bin/bash
set -e
cd /opt/ml-benchmark
/opt/ml-benchmark/venv/bin/python benchmark.py
RUN
  chmod +x /opt/ml-benchmark/run_benchmark.sh

  /opt/ml-benchmark/run_benchmark.sh

  cat >/opt/ml-benchmark/app.py <<'PY'
import json
import time

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

app = FastAPI(title="LightGBM CPU fallback inference")

started_at = time.time()
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42,
    stratify=data.target,
)

train_start = time.time()
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)
model.fit(X_train, y_train)
training_seconds = round(time.time() - train_start, 3)
positive_scores = model.predict_proba(X_test)[:, 1]
auc_roc = round(float(roc_auc_score(y_test, positive_scores)), 6)

sample_features = X_test[0].tolist()


class PredictRequest(BaseModel):
    features: Optional[List[float]] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "cpu-lightgbm",
        "training_seconds": training_seconds,
        "auc_roc": auc_roc,
    }


@app.get("/")
def root():
    return health()


@app.get("/metrics")
def metrics():
    return {
        "dataset": "sklearn breast_cancer",
        "rows": int(data.data.shape[0]),
        "features": int(data.data.shape[1]),
        "training_seconds": training_seconds,
        "auc_roc": auc_roc,
        "uptime_seconds": round(time.time() - started_at, 3),
    }


@app.post("/predict")
def predict(request: PredictRequest):
    features = request.features or sample_features
    array = np.asarray(features, dtype=float).reshape(1, -1)
    probability = float(model.predict_proba(array)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "model": "lightgbm-cpu-fallback",
        "prediction": prediction,
        "positive_probability": round(probability, 6),
    }

PY

  cat >/etc/systemd/system/lightgbm-api.service <<'SERVICE'
[Unit]
Description=LightGBM CPU fallback inference API
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/opt/ml-benchmark
ExecStart=/opt/ml-benchmark/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

  systemctl daemon-reload
  systemctl enable lightgbm-api
  systemctl start lightgbm-api
  echo "CPU LightGBM API started on port 8000"
fi
