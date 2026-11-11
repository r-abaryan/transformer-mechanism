# Sliding-window results

Executed and closed2026-10-02. All48 numerical cases,18 reachability cases and40 pair-count cases pass. No training was performed.

[Short guide and diagram](sliding-window-guide.md) | [Protocol](../../protocols/sliding-window.md)

## Methods and numerical findings

CPU float64, one thread, three seeds2411/2417/2423. Lengths5/9, radii0/1/2/n-1, causal and bidirectional. Batch2, heads2, Q/K width3, V width2. Inputs are independent random tensors. We compare dense masked softmax with explicit gathered-key arithmetic and stable exponentiation, then compare all Q/K/V gradients of a scalar probe. A joint central difference checks derivatives independently of autograd.

| Seed | Cases | Maximum output error | Maximum gradient error | Directional difference error |
|---|---:|---:|---:|---:|
| 2411 | 16 | 4.44e-16 | 1.39e-17 | 1.67e-11 |
| 2417 | 16 | 2.22e-16 | 2.43e-17 | 1.52e-11 |
| 2423 | 16 | 4.44e-16 | 1.39e-17 | 1.99e-11 |

Output/gradient tolerance1e-10; directional tolerance1e-7 at step1e-6. Self-only outputs equal V exactly. Full-radius outputs equal full attention exactly. This covers Q/K/V arithmetic, not learned projection gradients or a complete Transformer. All per-case differences from unrestricted attention are retained in the raw report.

| Deliberate error | Output change | Required minimum |
|---|---:|---:|
| Omit local window |2.00 |1e-6 |
| Permit a future key |3.33 |1e-6 |
| Exclude a boundary key |5.00 |1e-6 |

Controls use zero Q/K and a nonzero value at a chosen key. Uniform weights make the error visible without relying on random attention preferences. All three errors are detected.

## Reach and potential savings

Set propagation, adjacency-matrix powers and interval formulas agree in all18 configurations (length9, radius0/1/2, depth1/2/3, both policies). A separate uniform local-averaging stack has positive input/output derivatives on every reachable pair and exactly zero outside. This is a constructive witness of possible influence.

| Causal radius2, output position9 | Reachable original input positions |
|---|---|
| One layer |7 through9 |
| Two layers |5 through9 |
| Three layers |3 through9 |

Position1 cannot affect position9 within these three layers. This structural conclusion assumes only the tested local cross-token connections. It does not diagnose our earlier copying failures.

| Length | Radius | Policy | Local pairs | Full pairs | Local/full |
|---|---:|---|---:|---:|---:|
| 128 | 4 | Bidirectional | 1,132 | 16,384 | 6.91% |
| 128 | 4 | Causal | 630 | 8,256 | 7.63% |
| 1024 | 4 | Bidirectional | 9,196 | 1,048,576 | 0.88% |
| 1024 | 4 | Causal | 5,110 | 524,800 | 0.97% |
| 4096 | 4 | Bidirectional | 36,844 | 16,777,216 | 0.22% |
| 4096 | 4 | Causal | 20,470 | 8,390,656 | 0.24% |

Counts are per head, per example. All40 counts agree between row enumeration and closed formulas. They describe useful connections, not measured GPU memory or speed. Dense causal implementations can still compute a full square score matrix.

## Conclusion and stopping rule

Local attention preserves the tested masked arithmetic while restricting direct access. Depth expands possible access at a predictable rate. Neither reachability nor fewer pairs establishes trained quality or faster execution.

No numerical failures or threshold changes occurred. No padding, dropout, dilation, global tokens, GPU execution or training was tested. The study is closed; no training or implementation benchmark is automatically selected.

[Raw cases, controls and provenance](../../../results/sliding_window.json) | [Code](../../../experiments/sliding_window.py)
