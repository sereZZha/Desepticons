#!/usr/bin/env python3
"""T2: Safety Block для Unitree Go1 (MuJoCo Lowlevel), весь код в одном файле.

Идея: Judge шлёт команды в /lowcmd_bare, мы их проверяем и публикуем безопасные в /lowcmd.
Что проверяем:
  1. nan/inf в команде -> выбрасываем всю команду
  2. углы обрезаем по пределам суставов
  3. цель двигаем маленькими шагами (не больше 4 градусов за такт, в ТЗ можно до 5)
  4. не заходим в самоколлизию (проверяем через MuJoCo)
  5. если команды перестали приходить -> стоим на последней безопасной позе

Файл состоит из двух частей: логика проверок (SafetyCore, работает без ROS) и ROS 2 узел SafetyBlockNode.
Модель робота и список пар для самоколлизий берутся из папки models/unitree_go1 рядом с файлом
(или из переменной окружения GO1_MODELS_DIR).

Запуск (внутри контейнера с ROS 2 Humble и пакетом unitree_go):
    python3 safety_block.py --ros-args -p use_sim_time:=true
"""
import csv
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from unitree_go.msg import LowCmd, LowState
    HAVE_ROS = True
except ImportError:
    HAVE_ROS = False

N = 12
LEGS = ("FR", "FL", "RR", "RL")
JOINTS = ("hip", "thigh", "calf")
JOINT_NAMES = [f"{leg}_{j}_joint" for leg in LEGS for j in JOINTS]

Q_LO = np.array([-0.863, -0.686, -2.818] * 4)
Q_HI = np.array([0.863, 4.501, -0.888] * 4)
# Округляем границы q внутрь диапазона float32.
_Q_LO_F32 = Q_LO.astype(np.float32)
_Q_HI_F32 = Q_HI.astype(np.float32)
_Q_LO_F32 = np.where(_Q_LO_F32 < Q_LO, np.nextafter(_Q_LO_F32, np.float32(np.inf)), _Q_LO_F32)
_Q_HI_F32 = np.where(_Q_HI_F32 > Q_HI, np.nextafter(_Q_HI_F32, np.float32(-np.inf)), _Q_HI_F32)
STAND_Q = np.array([0.0, 0.8536, -1.7072] * 4)

MODE_SERVO = 0x0A  # Позиционное управление Go1.
POS_STOP = 1.0e8  # При |q| ≥ POS_STOP целевой угол не меняется.


@dataclass
class SafetyConfig:
    max_step_deg: float = 4.0  # Предел ТЗ — 5° на команду.
    check_step_deg: float = 0.1
    pen_margin: float = 0.0005  # Порог в метрах, с запасом до 1 мм.
    input_timeout: float = 0.3  # Симуляционное время, секунды.
    hold_kp: float = 40.0
    hold_kd: float = 2.0
    kp_min: float = 1.0
    kp_max: float = 100.0
    kd_min: float = 0.2
    kd_max: float = 5.0

    @property
    def max_step(self):
        return math.radians(self.max_step_deg)


def is_finite_command(raw):
    for key in raw:
        for el in raw[key]:
            if not np.isfinite(el):
                return False
    return True


def clamp_targets(q):
    n_q = np.empty(12)
    for i in range(12):
        n_q[i] = min(max(q[i], _Q_LO_F32[i]), _Q_HI_F32[i])
    return n_q.astype(np.float32).astype(float)


def apply_pos_stop(q, hold_q):
    res = np.empty(12)
    for i in range(12):
        if abs(q[i]) >= POS_STOP:
            res[i] = hold_q[i]
        else:
            res[i] = q[i]
    return res


def sanitize_gains(kp, kd, cfg):
    kp_new = np.empty(12)
    kd_new = np.empty(12)
    for i in range(12):
        if kp[i] < cfg.kp_min:
            kp_new[i] = cfg.hold_kp
        else:
            kp_new[i] = min(kp[i], cfg.kp_max)
        kd_new[i] = min(max(kd[i], cfg.kd_min), cfg.kd_max)
    return kp_new, kd_new


def step_towards(current, target, max_step):
    res = np.empty(12)
    for i in range(12):
        diff = target[i] - current[i]
        if diff > max_step:
            diff = max_step
        elif diff < -max_step:
            diff = -max_step
        res[i] = current[i] + diff
    return res


def is_input_lost(now, last_rx, timeout):
    if last_rx is None:
        return True
    return now - last_rx > timeout


class CollisionChecker:
    """Считает глубину проникновения ног друг в друга (по 655 парам из csv организаторов)."""

    def __init__(self, models_dir):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(os.path.join(models_dir, "scene.xml"))
        self.data = mujoco.MjData(self.model)
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)

        # Индексы qpos в порядке приводов LowCmd.
        self.qadr = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                              for n in JOINT_NAMES])
        ng = self.model.ngeom
        self.pair = np.zeros((ng, ng), dtype=bool)  # Пары, которые проверяет Judge.
        self.n_checked = 0
        with open(os.path.join(models_dir, "t2_collision_pairs.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["diagnostic_policy"] != "checked":
                    continue
                g1, g2 = int(row["geom1_id"]), int(row["geom2_id"])
                if not 0 <= g1 < g2 < ng or self.pair[g1, g2]:
                    raise ValueError(f"Некорректная пара геометрий: {g1}, {g2}")
                self.pair[g1, g2] = self.pair[g2, g1] = True
                self.n_checked += 1
        if self.n_checked != 655:
            raise ValueError(f"Ожидалось 655 проверяемых пар, получено {self.n_checked}")

    def penetration(self, q12):
        d = self.data
        d.qpos[self.qadr] = q12
        self.mj.mj_kinematics(self.model, d)
        self.mj.mj_collision(self.model, d)

        max_pen = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            # Отрицательный contact.dist означает проникновение геометрий.
            if self.pair[c.geom1, c.geom2] and c.dist < 0:
                if -c.dist > max_pen:
                    max_pen = -c.dist
        return float(max_pen)


class SafetyCore:
    """Основная логика: принимает команды и состояние, на каждом такте отдаёт безопасные q, kp, kd."""

    def __init__(self, cfg=None, checker=None):
        self.cfg = cfg or SafetyConfig()
        self.checker = checker  # В ROS проверка коллизий обязательна.
        self.out_q = clamp_targets(STAND_Q)
        self.target_q = self.out_q.copy()
        self.kp = np.full(N, self.cfg.hold_kp)
        self.kd = np.full(N, self.cfg.hold_kd)
        self.out_kp, self.out_kd = self.kp, self.kd
        self.last_rx = None
        self.got_cmd = False
        self.state_seen = False
        self.output_started = False
        self.pen_out = checker.penetration(self.out_q) if checker is not None else None
        if self.pen_out is not None and self.pen_out > self.cfg.pen_margin:
            raise ValueError("Начальная поза STAND_Q небезопасна")
        self.rejected = 0
        self.blocked = 0
        self.holding = True

    def on_state(self, q12):
        # Телеметрия задаёт начальную позу до первой публикации.
        q12 = np.asarray(q12, dtype=float)
        if self.state_seen or self.got_cmd or self.output_started or not np.all(np.isfinite(q12)):
            return
        candidate = clamp_targets(q12)
        pen = self.checker.penetration(candidate) if self.checker is not None else None
        if pen is not None and pen > self.cfg.pen_margin:
            return
        self.state_seen = True
        self.out_q = candidate
        self.target_q = self.out_q.copy()
        self.pen_out = pen

    def on_command(self, raw, now):
        self.last_rx = now  # Некорректная команда тоже считается полученным сообщением.

        if not is_finite_command(raw):
            self.rejected += 1
            self.target_q = self.out_q.copy()
            self.kp, self.kd = self.out_kp, self.out_kd
            return False

        q = apply_pos_stop(raw["q"][:12], self.out_q)
        self.target_q = clamp_targets(q)
        self.kp, self.kd = sanitize_gains(raw["kp"][:12], raw["kd"][:12], self.cfg)
        self.got_cmd = True
        return True

    def step(self, now):
        lost = is_input_lost(now, self.last_rx, self.cfg.input_timeout)
        self.holding = lost

        if lost:
            self.target_q = self.out_q.copy()
            kp = np.full(N, self.cfg.hold_kp)
            kd = np.full(N, self.cfg.hold_kd)
        else:
            kp = self.kp
            kd = self.kd

        # Проверяем углы после округления в float32.
        cand = step_towards(self.out_q, self.target_q, self.cfg.max_step).astype(np.float32).astype(float)
        if np.any(cand != self.out_q):
            if self._path_ok(self.out_q, cand):
                self.out_q = cand
            else:
                self.blocked += 1
                kp, kd = self.out_kp, self.out_kd

        self.output_started = True
        self.out_kp, self.out_kd = kp, kd
        return self.out_q.copy(), kp.copy(), kd.copy()

    def _path_ok(self, a, b):
        if self.checker is None:
            return True

        if self.pen_out is None:
            self.pen_out = self.checker.penetration(a)
        if self.pen_out > self.cfg.pen_margin:
            return False

        big = np.max(np.abs(b - a))
        n = max(1, int(math.ceil(big / math.radians(self.cfg.check_step_deg))))
        pen = 0.0
        for k in range(1, n + 1):
            pen = self.checker.penetration(a + (b - a) * k / n)
            if pen > self.cfg.pen_margin:
                return False

        self.pen_out = pen
        return True


if HAVE_ROS:
    def raw_from_lowcmd(msg):
        res = {}
        for key in ("q", "dq", "tau", "kp", "kd"):
            res[key] = np.array([getattr(m, key) for m in msg.motor_cmd], dtype=float)
        return res

    class SafetyBlockNode(Node):
        def __init__(self):
            super().__init__("safety_block")
            # Таймеры используют время симуляции из /clock.
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

            models_dir = os.environ.get(
                "GO1_MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "unitree_go1"))
            self.declare_parameter("models_dir", models_dir)
            self.declare_parameter("rate_hz", 100.0)

            checker = CollisionChecker(self.get_parameter("models_dir").value)
            self.get_logger().info(f"Проверка самоколлизий включена ({checker.n_checked} пар)")
            self.core = SafetyCore(SafetyConfig(), checker)

            # BEST_EFFORT на входе совместим с обоими режимами издателя.
            qos_in = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(LowCmd, "/lowcmd_bare", self.on_cmd, qos_in)
            self.create_subscription(LowState, "/lowstate", self.on_state,
                                     QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
            self.pub = self.create_publisher(LowCmd, "/lowcmd", 10)
            self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.tick)
            self.create_timer(5.0, self.report)
            self.get_logger().info("Safety Block запущен: /lowcmd_bare -> /lowcmd")

        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def on_state(self, msg):
            self.core.on_state([m.q for m in msg.motor_state[:N]])

        def on_cmd(self, msg):
            if not self.core.on_command(raw_from_lowcmd(msg), self.now()):
                self.get_logger().warn("Команда с NaN/Inf отклонена", throttle_duration_sec=2.0)

        def tick(self):
            q, kp, kd = self.core.step(self.now())
            msg = LowCmd()
            try:
                msg.head = [0xFE, 0xEF]
                msg.level_flag = 0xFF
            except Exception:
                pass
            for i in range(N):
                m = msg.motor_cmd[i]
                m.mode = MODE_SERVO
                m.q = float(q[i])
                m.dq = 0.0
                m.tau = 0.0
                m.kp = float(kp[i])
                m.kd = float(kd[i])
            self.pub.publish(msg)

        def report(self):
            c = self.core
            state = "УДЕРЖАНИЕ" if c.holding else "работа"
            self.get_logger().info(f"режим: {state}, отклонено NaN/Inf: {c.rejected}, "
                                   f"шагов заблокировано коллизией: {c.blocked}")


def main(args=None):
    if not HAVE_ROS:
        sys.exit("Не найдены rclpy / unitree_go: сделайте source /opt/ros/humble/setup.bash и соберите unitree_go")
    rclpy.init(args=args)
    node = SafetyBlockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
