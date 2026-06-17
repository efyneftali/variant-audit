# Infra

The deployment arc — local Docker now, Kubernetes later (your platform-eng layer).

## Now — Docker Compose (Weeks 1–3)
```bash
docker compose up -d qdrant     # Day 1: just the vector DB
docker compose up -d            # Week 3: + OTel collector, Tempo, Prometheus, Grafana
```
Files here: `docker-compose.yml` + `otel-collector-config.yaml`, `prometheus.yml`, `tempo.yaml`, `grafana-datasources.yaml`.

## Week 4 — local Kubernetes (`kind`)
Containerize the API + MCP server (multi-stage Dockerfiles, non-root), deploy to a
local `kind` cluster, install observability via the `kube-prometheus-stack` Helm chart,
add the nightly eval CronJob. → `k8s/` + `helm/` (to be added in Week 4).

## Optional — AWS EKS
Same Helm charts on a real cluster: GitHub Actions → ECR → Helm → EKS, IRSA for pod
permissions, ALB ingress, HPA on the stateless API. Tear it down when not demoing.
(In the cut-list — `kind` is enough to demo and talk about.)
