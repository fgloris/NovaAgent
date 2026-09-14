"""Pure Python pose validation/interpolation utilities."""

import math
from dataclasses import dataclass


@dataclass
class Pose:
    position: list
    orientation: list


def _finite(v):
    return all(math.isfinite(float(x)) for x in v)


def normalize(q):
    if not isinstance(q, list) or len(q) != 4 or not _finite(q):
        raise ValueError("invalid_pose")
    n = math.sqrt(sum(float(x) * float(x) for x in q))
    if n < 1e-9:
        raise ValueError("invalid_pose")
    return [float(x) / n for x in q]


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def rotate(q, v):
    return quat_mul(quat_mul(q, [*v, 0]), [-q[0], -q[1], -q[2], q[3]])[:3]


def compose(base, delta):
    r = rotate(base.orientation, delta.position)
    return Pose(
        [base.position[i] + r[i] for i in range(3)],
        normalize(quat_mul(base.orientation, delta.orientation)),
    )


def slerp(a, b, t):
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


def interpolate(poses, linear_speed=0.1, angular_speed=0.5, hz=20):
    if not poses:
        raise ValueError("empty_waypoints")
    out = []
    for a, b in zip(poses, poses[1:]):
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
                )
            )
    out.append(poses[-1])
    return out


def validate_waypoints(data, max_count=100):
    if not isinstance(data, list) or not data:
        raise ValueError("empty_waypoints")
    if len(data) > max_count:
        raise ValueError("too_many_waypoints")
    out = []
    for w in data:
        p = w.get("position") if isinstance(w, dict) else None
        if not isinstance(p, list) or len(p) != 3 or not _finite(p):
            raise ValueError("invalid_pose")
        out.append(Pose([float(x) for x in p], normalize(w.get("orientation"))))
    return out
