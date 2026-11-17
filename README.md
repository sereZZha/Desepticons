# Hackathon Go1 — Команда Desepticons

Решение для заочного этапа Hackathon Go1 (задание T1).

## Сборка Docker-образа

```bash
docker build --platform linux/amd64 -t go1-participant:latest .
docker run --rm --network none --entrypoint /entrypoint.sh go1-participant:latest ping
EDF
