#!/usr/bin/env bash
set -e

MODE="${1:-}"

case "$MODE" in
  ping)
    cat << 'JSON'
{
  "protocol_version": "1.0",
  "supported_tasks": ["T2"],
  "self_trained_rl": false,
  "preferred_speed_mps": null
}
JSON
    exit 0
    ;;
  T2)
    source /opt/ros/humble/setup.bash
    source /opt/unitree_ws/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    echo "Starting task T2..." >&2
    exec python3 /home/participant/safety_block.py
    ;;
  T3|T4|T5)
    source /opt/ros/humble/setup.bash
    source /opt/unitree_ws/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    echo "Starting task $MODE..." >&2
    exec sleep infinity
    ;;
  *)
    echo "Unknown argument: $MODE" >&2
    exit 1
    ;;
esac
