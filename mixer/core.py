"""mixer 核心 API：逐样本求和、增益、attack/release 限幅与流式处理。

设计要点
--------
- 单轨走快速路径，校验后原样返回，不做任何 DSP（逐样本不变）。
- 多轨固定分块（默认 4096 样本）流式求和，短轨补零，辅助内存与轨道
  总长度无关，可处理任意长的轨道。
- 限幅为宽带增益包络：attack 快速压低增益、release 缓慢恢复；每个样本
  再用安全增益 ``1/|x|`` 兜底，保证输出严格落在 ``[-1, 1]``，不产生削波。
- 限幅包络是每次混音调用内的局部状态，同一 Mixer 并发调用互不干扰；
  peak 记录用锁保护，并发结果与串行一致。
- 非法增益（非有限、负数、大于 1）抛 ``ValueError``；非有限样本抛
  ``FloatingPointError``，两类错误可区分。
"""

import math
from collections.abc import Sized
from itertools import islice
from threading import Lock

# 48kHz 下 attack 约 1ms、release 约 120ms 的一阶平滑系数。
_SAMPLE_RATE = 48000
_ATTACK_COEF = math.exp(-1.0 / (0.001 * _SAMPLE_RATE))
_RELEASE_COEF = math.exp(-1.0 / (0.120 * _SAMPLE_RATE))

_DEFAULT_CHUNK = 4096


class Mixer:
    def __init__(self):
        self._peak = 0.0
        self._lock = Lock()

    # ------------------------------------------------------------------ API
    def mix(self, tracks, gains):
        """混音并返回 ``list[float]``。

        输出长度为最长轨道长度，短轨尾部按零补齐。单轨原样返回；
        多轨逐样本加权求和后经 attack/release 限幅，保证不削波。
        """
        n, safe_gains, tracks = self._validate(tracks, gains)
        if n == 0:
            with self._lock:
                self._peak = 0.0
            return []

        # 单轨旧路径：不做增益乘法、不做限幅，逐样本原样输出。
        if n == 1:
            data = tracks[0]
            if not isinstance(data, list):
                data = list(data)
            if not all(map(math.isfinite, data)):
                raise FloatingPointError("non-finite sample in track")
            peak = 0.0
            for x in data:
                ax = x if x >= 0.0 else -x
                if ax > peak:
                    peak = ax
            with self._lock:
                self._peak = float(peak)
            return data

        out = []
        peak = 0.0
        for chunk, chunk_peak in self._engine(tracks, safe_gains, n,
                                              _DEFAULT_CHUNK):
            out.extend(chunk)
            if chunk_peak > peak:
                peak = chunk_peak
        with self._lock:
            self._peak = peak
        return out

    def mix_stream(self, tracks, gains, chunk_size=_DEFAULT_CHUNK):
        """流式混音生成器，逐块产出 ``(chunk, chunk_peak)``。

        tracks 中的每个轨道可以是列表/元组等定长序列，也可以是任意
        迭代器（包括生成器）；全程只保留分块大小的活动内存，适合
        一小时级长轨。输出拼接后与 :meth:`mix` 完全一致。
        """
        n, safe_gains, tracks = self._validate(tracks, gains)
        if n == 0:
            return
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")

        # 单轨旧路径在流式入口同样不做任何 DSP。
        if n == 1:
            src = tracks[0]
            if isinstance(src, Sized):
                iterator = iter(src)
            else:
                iterator = src
            peak = 0.0
            while True:
                chunk = list(islice(iterator, chunk_size))
                if not chunk:
                    break
                if not all(map(math.isfinite, chunk)):
                    raise FloatingPointError("non-finite sample in track")
                chunk_peak = 0.0
                for x in chunk:
                    ax = x if x >= 0.0 else -x
                    if ax > chunk_peak:
                        chunk_peak = ax
                if chunk_peak > peak:
                    peak = chunk_peak
                yield chunk, chunk_peak
            with self._lock:
                self._peak = float(peak)
            return

        peak = 0.0
        for chunk, chunk_peak in self._engine(tracks, safe_gains, n,
                                              chunk_size):
            if chunk_peak > peak:
                peak = chunk_peak
            yield chunk, chunk_peak
        with self._lock:
            self._peak = peak

    def peak(self):
        """返回最近一次混音输出的峰值（最大绝对值）。"""
        with self._lock:
            return self._peak

    def normalize(self, samples):
        """按峰值归一化，使最大绝对值恰好为 1.0。"""
        data = list(samples)
        if not all(map(math.isfinite, data)):
            raise FloatingPointError("non-finite sample in samples")
        peak = 0.0
        for x in data:
            ax = x if x >= 0.0 else -x
            if ax > peak:
                peak = ax
        if peak == 0.0:
            raise ValueError("cannot normalize a silent (zero) signal")
        factor = 1.0 / peak
        return [x * factor for x in data]

    # -------------------------------------------------------------- internals
    def _validate(self, tracks, gains):
        """校验轨道/增益，返回 (轨道数, 浮点增益列表, 定长轨道容器)。"""
        if not isinstance(tracks, Sized):
            tracks = tuple(tracks)
        n = len(tracks)
        if not isinstance(gains, Sized) or len(gains) != n:
            raise ValueError("gains length must match tracks length")
        safe_gains = []
        for g in gains:
            if not isinstance(g, (int, float)) or not math.isfinite(g) \
                    or g < 0.0 or g > 1.0:
                raise ValueError(
                    "gain must be a finite number within [0, 1]: %r" % (g,))
            safe_gains.append(float(g))
        return n, safe_gains, tracks

    def _engine(self, tracks, gains, n, chunk_size):
        """分块求和 + attack/release 限幅的流式引擎。"""
        isfinite = math.isfinite
        fabs = math.fabs
        ca = _ATTACK_COEF
        cr = _RELEASE_COEF

        sources = []
        positions = []
        for t in tracks:
            if isinstance(t, Sized):
                sources.append(t)
                positions.append(0)       # >=0 表示定长序列的读位置
            else:
                sources.append(iter(t))
                positions.append(-1)      # -1 表示纯迭代器
        alive = [True] * n
        env = 1.0                         # 限幅增益包络，调用内局部状态

        while True:
            acc = [0.0] * chunk_size
            used = 0

            for i in range(n):
                if not alive[i]:
                    continue
                src = sources[i]
                if positions[i] >= 0:
                    p = positions[i]
                    chunk = src[p:p + chunk_size]
                    positions[i] = p + len(chunk)
                    if len(chunk) < chunk_size:
                        alive[i] = False
                else:
                    chunk = list(islice(src, chunk_size))
                    if len(chunk) < chunk_size:
                        alive[i] = False
                if not chunk:
                    continue
                if len(chunk) > used:
                    used = len(chunk)

                if not all(map(isfinite, chunk)):
                    raise FloatingPointError(
                        "non-finite sample in track %d" % i)

                gi = gains[i]
                if gi != 0.0:
                    # 原地累加，保持 acc 始终为 chunk_size 长度，
                    # 不能用 zip 推导式重绑定（短轨块会把 acc 缩短）。
                    for k, x in enumerate(chunk):
                        acc[k] += gi * x

            if used == 0:
                return

            # 宽带增益限幅：attack 压增益、release 恢复，安全增益兜底。
            # 注意 acc 始终保持 chunk_size 长度（长轨数据不会因短轨耗尽
            # 而被截断），只把前 used 个有效样本作为本块结果产出。
            chunk_peak = 0.0
            for j in range(used):
                x = acc[j]
                ax = fabs(x)
                if ax > 1.0:
                    safe = 0.0 if not isfinite(ax) else 1.0 / ax
                else:
                    safe = 1.0

                if safe < env:
                    env = safe + (env - safe) * ca       # attack
                else:
                    env = safe + (env - safe) * cr       # release

                applied = safe if env > safe else env    # 严格不削波兜底
                y = x * applied
                if not isfinite(y):                      # 求和溢出的极端情况
                    y = math.copysign(1.0, x)
                acc[j] = y

                ay = fabs(y)
                if ay > chunk_peak:
                    chunk_peak = ay

            yield acc[:used], chunk_peak
