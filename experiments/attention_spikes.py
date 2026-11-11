"""Small attention experiments. No training or paper benchmark reproduction."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
from datetime import datetime, timezone


def transpose(a):
    return list(map(list, zip(*a, strict=True)))


def matmul(a, b):
    columns = transpose(b)
    return [[sum(x * y for x, y in zip(row, col, strict=True))
             for col in columns] for row in a]


def softmax(row):
    maximum = max(row)
    if not math.isfinite(maximum):
        raise ValueError("Softmax needs at least one finite visible score")
    values = [math.exp(x - maximum) for x in row]
    total = sum(values)
    return [x / total for x in values]


def attention(q, k, v, causal=False):
    scores = matmul(q, transpose(k))
    scale = math.sqrt(len(q[0]))
    weights = [softmax([s / scale if not causal or j <= i else -math.inf
                        for j, s in enumerate(row)])
               for i, row in enumerate(scores)]
    return matmul(weights, v), weights


def error(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b, strict=True)
               for x, y in zip(ra, rb, strict=True))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def scaling():
    measurements = []
    samples = 4000
    for seed in (11, 29, 47):
        rng = random.Random(seed)
        for width in (4, 32, 128):
            dots = [sum(rng.gauss(0, 1) * rng.gauss(0, 1)
                        for _ in range(width)) for _ in range(samples)]
            raw = statistics.variance(dots)
            scaled = raw / width
            require(abs(scaled - 1) < 0.15,
                    f"Scaled variance outside predeclared tolerance: {scaled}")
            require(raw > 2, "Unscaled control failed to differ from unit variance")
            measurements.append(dict(seed=seed, width=width, raw_variance=raw,
                                     scaled_variance=scaled))
    return dict(samples_per_case=samples, tolerance=0.15, cases=measurements)


def masking():
    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
    changed = x[:2] + [[9.0, -7.0], [8.0, 6.0]]
    before, weights = attention(x, x, x, causal=True)
    after, _ = attention(changed, changed, changed, causal=True)
    masked_error = error(before[:2], after[:2])
    unmasked_before, _ = attention(x, x, x)
    unmasked_after, _ = attention(changed, changed, changed)
    control = error(unmasked_before[:2], unmasked_after[:2])
    require(masked_error < 1e-12, "Future information leaked through causal mask")
    require(control > 0.1, "Future edit has no effect in the unmasked control")
    require(all(weights[i][j] == 0 for i in range(4) for j in range(i + 1, 4)),
            "Forbidden attention weight is nonzero")
    try:
        softmax([-math.inf, -math.inf])
    except ValueError:
        rejected = True
    else:
        rejected = False
    require(rejected, "All-masked row must be rejected")
    return dict(causal_error=masked_error, unmasked_change=control,
                all_masked_row_rejected=rejected)


def permutation():
    x = [[0.2, 1.0], [-0.3, 0.5], [1.2, -0.7]]
    order = [2, 0, 1]
    permuted = [x[i] for i in order]
    y, _ = attention(x, x, x)
    yp, _ = attention(permuted, permuted, permuted)
    equivariance_error = error(yp, [y[i] for i in order])
    positions = [[0, 0], [1, -1], [2, -2]]
    def add_positions(rows):
        return [[a + b for a, b in zip(row, pos, strict=True)]
                for row, pos in zip(rows, positions, strict=True)]
    xp, pp = add_positions(x), add_positions(permuted)
    y_pos, _ = attention(xp, xp, xp)
    perm_pos, _ = attention(pp, pp, pp)
    positional_change = error(perm_pos, [y_pos[i] for i in order])
    require(equivariance_error < 1e-12, "Permutation equivariance failed")
    require(positional_change > 0.1, "Position-dependent control did not differ")
    return dict(equivariance_error=equivariance_error,
                fixed_position_control_change=positional_change)


def gradient():
    scores = [0.3, -0.7, 1.1]
    p = softmax(scores)
    jacobian = [[p[i] * (int(i == j) - p[j]) for j in range(3)]
                for i in range(3)]
    step = 1e-6
    numeric_columns = []
    for j in range(3):
        plus, minus = scores.copy(), scores.copy()
        plus[j] += step
        minus[j] -= step
        numeric_columns.append([(a - b) / (2 * step)
                                for a, b in zip(softmax(plus), softmax(minus), strict=True)])
    numeric = transpose(numeric_columns)
    gradient_error = error(jacobian, numeric)
    wrong = [[p[i] if i == j else 0 for j in range(3)] for i in range(3)]
    wrong_error = error(wrong, numeric)
    saturated = softmax([30.0, 0.0, 0.0])
    require(gradient_error < 1e-8, "Analytic softmax derivative disagrees with finite differences")
    require(wrong_error > 0.01, "Incorrect derivative escaped the control")
    require(saturated[0] * (1 - saturated[0]) < 1e-10, "Saturation witness failed")
    return dict(step=step, max_gradient_error=gradient_error,
                wrong_jacobian_error=wrong_error,
                saturated_diagonal=saturated[0] * (1 - saturated[0]))


def nonunique():
    a, b, values = [0.2, 0.6, 0.2], [0.4, 0.2, 0.4], [[0.0], [1.0], [2.0]]
    out_a, out_b = matmul([a], values), matmul([b], values)
    recovered = softmax([math.log(x) for x in a])
    changed_values = [[0.0], [1.0], [3.0]]
    control = error(matmul([a], changed_values), matmul([b], changed_values))
    require(error(out_a, out_b) < 1e-12, "Non-unique attention witness failed")
    require(error([a], [recovered]) < 1e-12, "Attention row not recovered from log scores")
    require(control > 0.1, "Changed-value control failed")
    return dict(attention_a=a, attention_b=b, values=values, output_a=out_a,
                output_b=out_b, changed_value_difference=control)


def streaming_row(scores, values):
    maximum, denominator = -math.inf, 0.0
    numerator = [0.0] * len(values[0])
    for score, value in zip(scores, values, strict=True):
        new_maximum = max(maximum, score)
        old_scale = math.exp(maximum - new_maximum)
        new_scale = math.exp(score - new_maximum)
        numerator = [old_scale * a + new_scale * b
                     for a, b in zip(numerator, value, strict=True)]
        denominator = old_scale * denominator + new_scale
        maximum = new_maximum
    return [x / denominator for x in numerator]


def streaming():
    rng = random.Random(83)
    cases = []
    for length in (1, 7, 64):
        for magnitude in (1.0, 1000.0):
            scores = [rng.uniform(-magnitude, magnitude) for _ in range(length)]
            values = [[rng.uniform(-2, 2) for _ in range(3)] for _ in range(length)]
            dense = matmul([softmax(scores)], values)
            online = [streaming_row(scores, values)]
            difference = error(dense, online)
            require(difference < 1e-12, "Online accumulation differs from dense result")
            cases.append(dict(length=length, score_bound=magnitude, max_error=difference))
    return dict(seed=83, cases=cases,
                limitation="Scalar online softmax identity; not a FlashAttention implementation")


def associativity():
    q, k, v = [[1.0, 0.0], [0.0, 2.0]], [[1.0, 2.0], [-1.0, 1.0]], [[2.0], [-1.0]]
    direct = matmul(matmul(q, transpose(k)), v)
    reordered = matmul(q, matmul(transpose(k), v))
    soft = matmul([softmax(row) for row in matmul(q, transpose(k))], v)
    raw_error, soft_difference = error(direct, reordered), error(soft, reordered)
    require(raw_error < 1e-12, "Raw matrix associativity failed")
    require(soft_difference > 0.1, "Softmax counterexample failed")
    return dict(raw_associativity_error=raw_error, softmax_reordering_difference=soft_difference)


def storage():
    lengths = (16, 32, 64)
    counts = [sum(len(row) for row in [[0.0] * n for _ in range(n)]) for n in lengths]
    ratios = [b / a for a, b in zip(counts[:-1], counts[1:], strict=True)]
    require(ratios == [4.0, 4.0], "Dense element count is not quadratic")
    return dict(lengths=lengths, score_elements=counts, doubling_ratios=ratios,
                limitation="Element counts only; no byte allocation or runtime benchmark")


EXPERIMENTS = [("S01", scaling), ("S02", masking), ("S03", permutation),
               ("S04", gradient), ("S05", nonunique), ("S06", streaming),
               ("S07", associativity), ("S08", storage)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/attention_spikes.json"))
    args = parser.parse_args()
    records = []
    for identifier, experiment in EXPERIMENTS:
        try:
            metrics = experiment()
            records.append(dict(id=identifier, name=experiment.__name__, status="pass", metrics=metrics))
        except Exception as exc:
            records.append(dict(id=identifier, name=experiment.__name__, status="fail",
                                error=f"{type(exc).__name__}: {exc}"))
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, platform=platform.platform(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  arithmetic="Python float (binary64), CPU, standard library",
                  records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for record in records:
        print(f"{record['id']} {record['name']}: {record['status']}")
        if record["status"] == "fail":
            print(record["error"])
    print(f"Report: {args.output}")
    return int(any(record["status"] != "pass" for record in records))


if __name__ == "__main__":
    sys.exit(main())
