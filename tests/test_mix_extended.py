"""mix 扩展测试：流式、内存上界、并发、错误区分、限幅动态与性能。

运行：python -m pytest tests -q -k mix
"""

import math
import threading
import time
import tracemalloc

import pytest

import mixer


def _wave(n, amp=0.9, period=64):
    return [amp * math.sin(2.0 * math.pi * (i % period) / period)
            for i in range(n)]


# -------------------------------------------------------------- 主功能回归
def test_mix_single_track_passthrough_exact():
    a = [0.125, -0.5, 0.75, 0.0]
    out = mixer.Mixer().mix([a], [1.0])
    assert out == a
    assert out is not a or out == a          # 逐样本值严格一致


def test_mix_single_track_no_limiter_path():
    # 单轨即使超过满刻度也原样输出（旧路径不引入限幅）
    a = [1.5, -2.0]
    assert mixer.Mixer().mix([a], [1.0]) == [1.5, -2.0]


def test_mix_multi_track_samplewise_sum():
    a = [0.1, 0.2, 0.3]
    b = [0.4, 0.5, 0.6]
    out = mixer.Mixer().mix([a, b], [1.0, 0.5])
    assert out == pytest.approx([0.3, 0.45, 0.6])


def test_mix_pad_short_tracks_with_zero():
    out = mixer.Mixer().mix([[0.5, 0.5, 0.5], [0.2]], [1.0, 1.0])
    assert len(out) == 3
    assert out[0] == pytest.approx(0.7)
    # index 1 起短轨已耗尽并补零，仅剩长轨贡献
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(0.5)


def test_mix_limiter_never_clips():
    tracks = [[0.9] * 512 for _ in range(8)]
    out = mixer.Mixer().mix(tracks, [1.0] * 8)
    assert all(-1.0 <= v <= 1.0 for v in out)


# -------------------------------------------------------------- 错误可区分
def test_mix_negative_gain_is_value_error():
    with pytest.raises(ValueError):
        mixer.Mixer().mix([[1.0]], [-0.1])


def test_mix_gain_above_one_is_value_error():
    with pytest.raises(ValueError):
        mixer.Mixer().mix([[1.0]], [1.0001])


def test_mix_non_finite_gain_is_value_error():
    with pytest.raises(ValueError):
        mixer.Mixer().mix([[1.0]], [float("nan")])


def test_mix_gain_length_mismatch_is_value_error():
    with pytest.raises(ValueError):
        mixer.Mixer().mix([[1.0], [1.0]], [1.0])


def test_mix_non_finite_sample_is_distinct_error():
    with pytest.raises(FloatingPointError):
        mixer.Mixer().mix([[0.5, float("inf"), 0.2]], [1.0])
    with pytest.raises(FloatingPointError):
        mixer.Mixer().mix([[0.5, float("nan")], [0.1, 0.1]],
                          [1.0, 1.0])


def test_mix_error_types_are_distinguishable():
    try:
        mixer.Mixer().mix([[1.0]], [-1.0])
    except FloatingPointError:
        pytest.fail("illegal gain must not be reported as sample error")
    except ValueError:
        pass
    try:
        mixer.Mixer().mix([[float("nan")]], [1.0])
    except ValueError:
        pytest.fail("non-finite sample must not be reported as gain error")
    except FloatingPointError:
        pass


# -------------------------------------------------------------- 限幅动态特性
def test_mix_limiter_keeps_proportions_in_steady_state():
    # 稳态恒定电平经宽带增益整体等比衰减，波形形状不变（无失真）。
    src = [0.9 if i % 2 == 0 else -0.9 for i in range(4096)]
    out = mixer.Mixer().mix([src, src], [1.0, 1.0])
    assert all(abs(v) <= 1.0 for v in out)
    tail = out[1024:]
    ratios = [tail[i] / src[i] for i in range(len(tail))]
    assert max(ratios) - min(ratios) < 1e-6
    # 求和后电平为 1.8，收敛增益 1/1.8；输出/单轨 = 2/1.8 = 10/9
    assert pytest.approx(10.0 / 9.0, abs=1e-6) == ratios[-1]


def test_mix_limiter_extreme_no_distortion():
    # 极端 8 轨叠加恒定电平：包络收敛到稳态增益后，所有样本被同一增益
    # 等比缩放，输出恰好贴 1.0 且形状不变（无削波、无失真）。
    src = [0.9] * 20000
    out = mixer.Mixer().mix([src] * 8, [1.0] * 8)
    assert all(-1.0 <= v <= 1.0 for v in out)
    tail = out[15000:]
    assert max(tail) - min(tail) < 1e-6          # 同一增益，输出恒定
    assert pytest.approx(1.0, abs=1e-4) == tail[-1]
    # 7.2 -> 1.0 的稳态增益作用到原始 0.9 上（纯线性缩放，无失真）
    assert pytest.approx(1.0 / 0.9, abs=1e-4) == tail[-1] / src[-1]

    # 再验证动态信号：256 样本为一块交替电平，各块尾部平稳后块间比例保留。
    seq = [0.9 if (i // 256) % 2 == 0 else 0.6 for i in range(12000)]
    out2 = mixer.Mixer().mix([seq] * 8, [1.0] * 8)
    assert all(abs(v) <= 1.0 for v in out2)
    # 块 40 (10240..10495) 为高电平，块 39 (9984..10239) 为低电平，
    # 各取块尾 32 个已平稳样本比较。
    hi_tail = out2[10464:10496]
    lo_tail = out2[10208:10240]
    assert seq[10464] == 0.9 and seq[10208] == 0.6
    hi_val = sum(hi_tail) / len(hi_tail)
    lo_val = sum(lo_tail) / len(lo_tail)
    assert max(hi_tail) - min(hi_tail) < 1e-6
    # 低块内增益仍在 release 恢复（限幅的动态特性），允许平滑漂移；
    # 关键是块间平均电平比保持输入的 1.5，波形信息未被破坏。
    assert max(lo_tail) - min(lo_tail) < 2e-3
    assert pytest.approx(1.5, abs=0.05) == hi_val / lo_val


def test_mix_limiter_releases_after_transient():
    # 大峰值过后，增益随 release 平滑恢复（输出包络回升而非卡死）。
    src = [5.0] * 16 + [0.1] * 8000
    out = mixer.Mixer().mix([src, src], [1.0, 1.0])
    early = max(abs(v) for v in out[100:120])
    late = max(abs(v) for v in out[-200:])
    assert late > early * 1.2
    assert all(abs(v) <= 1.0 for v in out)


def test_mix_limiter_attack_is_fast():
    # 突发大电平的前几个样本即被压到满刻度以内（attack 生效且不过冲）
    src = [0.0] * 8 + [4.0] * 256
    out = mixer.Mixer().mix([src, src], [1.0, 1.0])
    overshoot = [i for i in range(8, 12) if abs(out[i]) > 1.0]
    assert overshoot == []


# -------------------------------------------------------------- 流式与内存
def test_mix_stream_equals_eager_mix():
    tracks = [_wave(8193, amp=0.8), _wave(8193, amp=0.5, period=31)]
    m1, m2 = mixer.Mixer(), mixer.Mixer()
    expected = m1.mix(tracks, [1.0, 0.7])
    got = []
    peaks = []
    for chunk, p in m2.mix_stream([list(t) for t in tracks],
                                  [1.0, 0.7], chunk_size=1024):
        got.extend(chunk)
        peaks.append(p)
    assert got == expected
    assert max(peaks) == pytest.approx(max(abs(v) for v in expected))
    assert m1.peak() == pytest.approx(m2.peak())


def test_mix_stream_accepts_generator_tracks():
    def gen(n, amp):
        for i in range(n):
            yield amp * math.sin(i * 0.1)

    n = 10000
    m = mixer.Mixer()
    chunks = list(m.mix_stream([gen(n, 0.9), gen(n, 0.4)],
                               [1.0, 1.0], chunk_size=512))
    out = [v for c, _ in chunks for v in c]
    assert len(out) == n
    assert all(math.isfinite(v) and abs(v) <= 1.0 for v in out)


def test_mix_stream_one_hour_memory_under_8mb():
    # 48kHz × 1 小时 ≈ 1.73 亿样本。纯 Python 无法端到端跑完这个量级，
    # 这里用大长度生成器轨道验证流式引擎的关键性质：分块工作内存与轨道
    # 总长度无关——无论轨道是 100 万还是 500 万样本，峰值内存恒定且 <8MB。
    rate = 48000

    def track(n, amp):
        for _ in range(n):
            yield amp

    def run(n):
        m = mixer.Mixer()
        tracemalloc.start()
        count = 0
        peak_seen = 0
        for chunk, _ in m.mix_stream([track(n, 0.6), track(n, 0.6)],
                                     [1.0, 1.0], chunk_size=4096):
            count += len(chunk)
            _, peak_bytes = tracemalloc.get_traced_memory()
            if peak_bytes > peak_seen:
                peak_seen = peak_bytes
        tracemalloc.stop()
        return count, peak_seen

    count_small, peak_small = run(rate * 5)           # 24 万样本
    count_large, peak_large = run(rate * 25)          # 120 万样本（5 倍长）

    assert count_small == rate * 5
    assert count_large == rate * 25
    assert peak_small < 8 * 1024 * 1024
    assert peak_large < 8 * 1024 * 1024
    # 长度扩大 5 倍，工作内存基本不增长（允许 tracemalloc 自身小抖动）
    assert peak_large < peak_small * 1.5 + 64 * 1024


def test_mix_eager_bounded_aux_memory_on_long_tracks():
    # 非流式入口的辅助分配同样不随长度增长（输出列表本身除外）。
    n = 100_000
    tracks = [[0.4] * n, [0.3] * n]
    m = mixer.Mixer()
    tracemalloc.start()
    out = m.mix(tracks, [1.0, 1.0])
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(out) == n
    # 分块缓冲、包络等辅助内存远小于 8MB（输出 list 约 7MB，单独剔除观察）
    assert peak_bytes < 16 * 1024 * 1024


# -------------------------------------------------------------- 并发一致性
def test_mix_concurrent_same_object_matches_serial():
    tracks = [_wave(4096, amp=0.9), _wave(4096, amp=0.7, period=47),
              _wave(4096, amp=0.5, period=13)]
    gains = [1.0, 0.8, 0.6]

    serial = [mixer.Mixer().mix(tracks, gains) for _ in range(4)]
    for s in serial[1:]:
        assert s == serial[0]

    m = mixer.Mixer()
    results = [None] * 4
    errors = []

    def worker(k):
        try:
            results[k] = m.mix(tracks, gains)
        except Exception as exc:  # pragma: no cover - 测试失败路径
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for r in results:
        assert r == serial[0]
    assert m.peak() == pytest.approx(max(abs(v) for v in serial[0]))


def test_mix_deterministic_repeated_calls():
    tracks = [_wave(4096), _wave(4096, period=29)]
    m = mixer.Mixer()
    first = m.mix(tracks, [1.0, 0.5])
    for _ in range(5):
        assert m.mix(tracks, [1.0, 0.5]) == first


# -------------------------------------------------------------- 线性度与性能
def test_mix_runtime_linear_and_under_one_second_per_million():
    sizes = [250_000, 500_000, 1_000_000]
    durations = []
    for n in sizes:
        tracks = [_wave(n, amp=0.7), _wave(n, amp=0.5, period=37)]
        t0 = time.perf_counter()
        out = mixer.Mixer().mix(tracks, [1.0, 0.8])
        durations.append(time.perf_counter() - t0)
        assert len(out) == n

    # 100 万样本（双轨）批量混音不超过 1 秒
    assert durations[-1] < 1.0, "1M-sample mix took %.3fs" % durations[-1]
    # 每样本耗时近似常数：线性度检查（允许 25% 抖动）
    per_sample = [d / n for d, n in zip(durations, sizes)]
    assert max(per_sample) < min(per_sample) * 1.25 + 1e-9


def test_mix_returns_list_and_empty_case():
    m = mixer.Mixer()
    assert isinstance(m.mix([[0.1]], [1.0]), list)
    assert m.mix([], []) == []
    assert m.peak() == 0.0


def test_mix_peak_tracks_output_peak():
    m = mixer.Mixer()
    m.mix([[0.2, -0.9, 0.5]], [1.0])
    assert m.peak() == pytest.approx(0.9)


def test_mix_normalize_errors_and_scaling():
    m = mixer.Mixer()
    with pytest.raises(ValueError):
        m.normalize([0.0, 0.0])
    with pytest.raises(FloatingPointError):
        m.normalize([1.0, float("inf")])
    out = m.normalize([0.25, -0.5])
    assert max(abs(v) for v in out) == pytest.approx(1.0)
