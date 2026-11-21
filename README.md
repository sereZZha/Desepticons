# Hackathon Go1 — Команда Desepticons

Решение для заочного этапа Hackathon Go1 (задания T1 и T2).

## Сборка Docker-образа

```bash
docker build --platform linux/amd64 -t go1-participant:latest .
docker run --rm --network none --entrypoint /entrypoint.sh go1-participant:latest ping
```

## Запуск T2

В репозитории симулятора, после загрузки образа Simulator, укажите его тег:

```bash
SIM_IMAGE=registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0
python3 run_local.py --task T2 --sim-image "$SIM_IMAGE" --participant-image go1-participant:latest --headless --logs
```

Safety Block читает `/lowcmd_bare` и публикует проверенные команды в `/lowcmd`.
