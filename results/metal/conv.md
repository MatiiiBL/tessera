# Metal: ResNet-50 convolutions

**Provisional:** measured while the machine was busy (load average 6.62 at the start). Re-run on a quiet machine for final numbers.

Measured 2026-11-12T06:54:14Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `cb94edd`. Reproduce with `uv run --group bench python -m tessera.backend.metal.bench conv`.

Each distinct convolution group of ResNet-50 at 224 x 224, batch 1, float32, with its fused bias,
residual addition and ReLU. Cells are GPU microseconds per call (GFLOPS in parentheses); MPS
runs the same operators with PyTorch 2.15.0 on channels-last tensors.

| Layer (NHWC input -> channels, kernel/stride) | Count | Tessera default | MPS |
|---|---|---|---|
| 224x224x3 -> 64, 7x7/2 +conv2d+relu | 1 | 1787 (132) | 321 (735) |
| 56x56x64 -> 64, 1x1/1 +relu | 1 | 127 (202) | 84 (306) |
| 56x56x64 -> 64, 3x3/1 +relu | 3 | 1747 (132) | 148 (1565) |
| 56x56x64 -> 256, 1x1/1 | 1 | 434 (237) | 151 (679) |
| 56x56x64 -> 256, 1x1/1 +add+relu | 3 | 450 (229) | 239 (430) |
| 56x56x256 -> 64, 1x1/1 +relu | 2 | 512 (201) | 135 (760) |
| 56x56x256 -> 128, 1x1/1 +relu | 1 | 995 (207) | 204 (1005) |
| 56x56x128 -> 128, 3x3/2 +relu | 1 | 1764 (131) | 228 (1015) |
| 56x56x256 -> 512, 1x1/2 | 1 | 932 (220) | 180 (1145) |
| 28x28x128 -> 512, 1x1/1 +add+relu | 4 | 452 (227) | 164 (626) |
| 28x28x512 -> 128, 1x1/1 +relu | 3 | 484 (212) | 160 (643) |
| 28x28x128 -> 128, 3x3/1 +relu | 3 | 1874 (123) | 208 (1114) |
| 28x28x512 -> 256, 1x1/1 +relu | 1 | 937 (219) | 241 (851) |
| 28x28x256 -> 256, 3x3/2 +relu | 1 | 1782 (130) | 297 (778) |
| 28x28x512 -> 1024, 1x1/2 | 1 | 960 (214) | 215 (954) |
| 14x14x256 -> 1024, 1x1/1 +add+relu | 6 | 437 (235) | 150 (684) |
| 14x14x1024 -> 256, 1x1/1 +relu | 5 | 531 (193) | 169 (608) |
| 14x14x256 -> 256, 3x3/1 +relu | 5 | 1980 (117) | 182 (1268) |
| 14x14x1024 -> 512, 1x1/1 +relu | 1 | 758 (271) | 192 (1068) |
| 14x14x512 -> 512, 3x3/2 +relu | 1 | 2368 (98) | 979 (236) |
| 14x14x1024 -> 2048, 1x1/2 | 1 | 1676 (123) | 378 (543) |
| 7x7x512 -> 2048, 1x1/1 +add+relu | 3 | 554 (186) | 205 (500) |
| 7x7x2048 -> 512, 1x1/1 +relu | 2 | 660 (156) | 246 (417) |
| 7x7x512 -> 512, 3x3/1 +relu | 2 | 2576 (90) | 279 (829) |

Over the 53 convolutions of the network: Tessera 54.33 ms with default configurations; MPS 10.99 ms.
