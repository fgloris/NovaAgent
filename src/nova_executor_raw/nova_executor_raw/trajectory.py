"""纯 Python 位姿校验与插值工具。"""

import json
import math
from dataclasses import dataclass


@dataclass
class Pose:
    """表示笛卡尔位置和四元数姿态。"""
    position: list
    orientation: list
    gripper: float | None = None


def _finite(v):
    """判断序列中的所有数值是否为有限数。"""
    return all(math.isfinite(float(x)) for x in v)


def normalize(q):
    """归一化四元数，输入非法时抛出 invalid_pose。"""
    if not isinstance(q, list) or len(q) != 4 or not _finite(q):
        raise ValueError("invalid_pose")
    n = math.sqrt(sum(float(x) * float(x) for x in q))
    if n < 1e-9:
        raise ValueError("invalid_pose")
    return [float(x) / n for x in q]


def quat_mul(a, b):
    """按 XYZW 顺序计算两个四元数的乘积(Hamilton 积),结果表示先 b 后 a 的复合旋转。"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def rotate(q, v):
    """用单位四元数 q 旋转三维向量 v(计算 q * [v,0] * q⁻¹,取虚部)。

    四元数共轭即逆(单位四元数),故 [-x,-y,-z,w] 就是 q⁻¹。
    """
    return quat_mul(quat_mul(q, [*v, 0]), [-q[0], -q[1], -q[2], q[3]])[:3]


def compose(base, delta):
    """把相对位姿增量合成到基准位姿上,返回新的绝对位姿。

    位置:基准位置 + 基准姿态旋转后的增量位移;
    姿态:基准姿态与增量姿态的四元数乘积(先 base 再 delta)。
    """
    r = rotate(base.orientation, delta.position)
    return Pose(
        [base.position[i] + r[i] for i in range(3)],
        normalize(quat_mul(base.orientation, delta.orientation)),
        delta.gripper if delta.gripper is not None else base.gripper,
    )


def slerp(a, b, t):
    """在两个四元数之间做球面线性插值(SLERP),t=0 取 a、t=1 取 b。

    点积为负时先翻转 b 以走短弧;夹角极小时退化为线性插值避免除零。
    """
    a, b = normalize(a), normalize(b)
    d = sum(x * y for x, y in zip(a, b))
    if d < 0:
        b = [-x for x in b]
        d = -d
    if d > 0.9995:
        return normalize([a[i] + t * (b[i] - a[i]) for i in range(4)])
    th = math.acos(max(-1, min(1, d)))
    s = math.sin(th)
    return [
        (math.sin((1 - t) * th) * a[i] + math.sin(t * th) * b[i]) / s for i in range(4)
    ]


def interpolate(poses, linear_speed=0.1, angular_speed=0.5, gripper_speed=1.0, hz=20):
    """在相邻路径点之间插值,生成位置线性、姿态 SLERP、夹爪渐变的离散轨迹点。

    每段先完成位置/姿态运动(时长取"位移/线速度""转角/角速度""1/hz"最大值),
    再在到达位置后用"夹爪行程/夹爪速度"的时间完成夹爪开合,保证先到位再闭合。
    未指定夹爪的路径点沿用最近一次显式值(含起点)。

    返回 (轨迹点列表, 每点所属段索引);段 s 表示 poses[s] -> poses[s+1],
    故段 s 对应第 s+1 个 waypoint(poses[0] 为起点)。
    """
    if not poses:
        raise ValueError("empty_waypoints")
    out = []
    segments = []
    grip = poses[0].gripper
    for seg_index, (a, b) in enumerate(zip(poses, poses[1:])):
        start_len = len(out)
        target = b.gripper if b.gripper is not None else grip
        # 运动阶段:位置线性、姿态 slerp,夹爪保持段起点值
        d = math.sqrt(sum((b.position[i] - a.position[i]) ** 2 for i in range(3)))
        qd = abs(sum(x * y for x, y in zip(a.orientation, b.orientation)))
        ang = 2 * math.acos(max(-1, min(1, qd)))
        dur = max(d / max(linear_speed, 1e-6), ang / max(angular_speed, 1e-6), 1 / hz)
        n = max(1, int(math.ceil(dur * hz)))
        for j in range(n):
            t = j / n
            out.append(
                Pose(
                    [
                        a.position[k] + t * (b.position[k] - a.position[k])
                        for k in range(3)
                    ],
                    slerp(a.orientation, b.orientation, t),
                    grip,
                )
            )
        # 夹爪阶段:位置/姿态停在 b,夹爪从起点值线性过渡到目标值,末尾补一帧 dwell
        if grip is not None and abs(target - grip) > 1e-9:
            steps = max(1, int(math.ceil(abs(target - grip) / max(gripper_speed, 1e-6) * hz)))
            for j in range(1, steps + 1):
                t = j / steps
                out.append(
                    Pose(
                        list(b.position),
                        list(b.orientation),
                        grip + t * (target - grip),
                    )
                )
            out.append(Pose(list(b.position), list(b.orientation), target))
        segments.extend([seg_index] * (len(out) - start_len))
        grip = target
    last = poses[-1]
    out.append(Pose(list(last.position), list(last.orientation), grip))
    segments.append(max(0, len(poses) - 2))
    return out, segments


def _fmt_vec(values):
    """把数值序列格式化成 [a, b, c] 形式,便于放进面向调用方的错误信息。"""
    return "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"


def _parse_error(error):
    """把错误负载解析成 dict:已是 dict 直接用,JSON 字符串则解析,其余返回 None。"""
    if isinstance(error, dict):
        return error
    if isinstance(error, str):
        try:
            parsed = json.loads(error)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _hint(payload, *keys):
    """把指定诊断字段格式化成 ' (k=v, ...)';无可用字段时返回空串。"""
    pairs = [f"{key}={payload[key]}" for key in keys if key in payload]
    return f" ({', '.join(pairs)})" if pairs else ""


def describe_failure(error, segments, poses):
    """把后端结构化的失败信息渲染成面向调用方的中文说明。

    error 为 SimError 序列化后的 JSON 字符串(或 dict),形如
    '{"code": "ik_failed", "index": 3, "pos_err": 0.05, "rot_err": 0.03}';
    segments 为 interpolate 返回的轨迹点->段映射,poses 为插值前的路径点。
    返回 {message, failed_waypoint, failed_segment, diagnostics},无法识别时返回 None。
    """
    payload = _parse_error(error)
    if not payload:
        return None
    code, index = payload.get("code"), payload.get("index")
    if code not in ("ik_failed", "joint_jump_violation") or not isinstance(index, int):
        return None
    if not segments or index < 0 or index >= len(segments):
        return None
    seg = segments[index]
    target_index = seg + 1
    if target_index >= len(poses):
        return None
    target = poses[target_index]
    where = f"第{target_index}个waypoint {_fmt_vec(target.position)} {_fmt_vec(target.orientation)}"
    if code == "ik_failed":
        message = f"{where} 由于ik failed不可达{_hint(payload, 'pos_err', 'rot_err')}"
    else:
        message = f"第{target_index}个waypoint前发生了关节跳变{_hint(payload, 'jump')}"
    return {
        "message": message,
        "failed_waypoint": target_index,
        "failed_segment": seg,
        "diagnostics": payload,
    }


def validate_waypoints(data, max_count=100):
    """校验绝对路径点 JSON，并返回已归一化的 Pose 对象。"""
    if not isinstance(data, list) or not data:
        raise ValueError("empty_waypoints")
    if len(data) > max_count:
        raise ValueError("too_many_waypoints")
    out = []
    for w in data:
        p = w.get("position") if isinstance(w, dict) else None
        if not isinstance(p, list) or len(p) != 3 or not _finite(p):
            raise ValueError("invalid_pose")
        gripper = w.get("gripper")
        if gripper is not None and not math.isfinite(float(gripper)):
            raise ValueError("invalid_gripper")
        out.append(
            Pose(
                [float(x) for x in p],
                normalize(w.get("orientation")),
                None if gripper is None else float(gripper),
            )
        )
    return out
