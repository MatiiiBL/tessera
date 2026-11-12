# Metal: ResNet-50 convolutions

Measured 2026-11-12T07:38:49Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `fac1e9b`. Reproduce with `uv run --group bench python -m tessera.backend.metal.bench conv --trials 16`.

Each distinct convolution group of ResNet-50 at 224 x 224, batch 1, float32, with its fused bias,
residual addition and ReLU. Cells are GPU microseconds per call (GFLOPS in parentheses); MPS
runs the same operators with PyTorch 2.15.0 on channels-last tensors.

| Layer (NHWC input -> channels, kernel/stride) | Count | Tessera default | Tessera best of 16 | MPS |
|---|---|---|---|---|
| 224x224x3 -> 64, 7x7/2 +conv2d+relu | 1 | 1683 (140) | 867 (272) | 269 (878) |
| 56x56x64 -> 64, 1x1/1 +relu | 1 | 25 (1045) | 24 (1051) | 59 (437) |
| 56x56x64 -> 64, 3x3/1 +relu | 3 | 217 (1067) | 217 (1067) | 105 (2200) |
| 56x56x64 -> 256, 1x1/1 | 1 | 82 (1246) | 82 (1246) | 97 (1059) |
| 56x56x64 -> 256, 1x1/1 +add+relu | 3 | 106 (972) | 106 (972) | 224 (460) |
| 56x56x256 -> 64, 1x1/1 +relu | 2 | 116 (889) | 91 (1128) | 121 (846) |
| 56x56x256 -> 128, 1x1/1 +relu | 1 | 161 (1273) | 161 (1273) | 175 (1175) |
| 56x56x128 -> 128, 3x3/2 +relu | 1 | 2035 (114) | 1079 (214) | 210 (1100) |
| 56x56x256 -> 512, 1x1/2 | 1 | 970 (212) | 851 (241) | 154 (1332) |
| 28x28x128 -> 512, 1x1/1 +add+relu | 4 | 98 (1053) | 97 (1064) | 142 (723) |
| 28x28x512 -> 128, 1x1/1 +relu | 3 | 89 (1152) | 89 (1152) | 103 (1001) |
| 28x28x128 -> 128, 3x3/1 +relu | 3 | 201 (1151) | 201 (1151) | 124 (1863) |
| 28x28x512 -> 256, 1x1/1 +relu | 1 | 167 (1231) | 167 (1231) | 185 (1113) |
| 28x28x256 -> 256, 3x3/2 +relu | 1 | 1998 (116) | 1085 (213) | 287 (806) |
| 28x28x512 -> 1024, 1x1/2 | 1 | 1004 (205) | 872 (236) | 166 (1241) |
| 14x14x256 -> 1024, 1x1/1 +add+relu | 6 | 68 (1520) | 68 (1520) | 144 (713) |
| 14x14x1024 -> 256, 1x1/1 +relu | 5 | 80 (1286) | 80 (1286) | 121 (852) |
| 14x14x256 -> 256, 3x3/1 +relu | 5 | 265 (872) | 225 (1026) | 149 (1547) |
| 14x14x1024 -> 512, 1x1/1 +relu | 1 | 139 (1474) | 139 (1474) | 182 (1129) |
| 14x14x512 -> 512, 3x3/2 +relu | 1 | 2800 (83) | 1363 (170) | 891 (260) |
| 14x14x1024 -> 2048, 1x1/2 | 1 | 1763 (117) | 1207 (170) | 334 (616) |
| 7x7x512 -> 2048, 1x1/1 +add+relu | 3 | 80 (1278) | 80 (1278) | 192 (536) |
| 7x7x2048 -> 512, 1x1/1 +relu | 2 | 93 (1102) | 93 (1102) | 183 (563) |
| 7x7x512 -> 512, 3x3/1 +relu | 2 | 308 (750) | 308 (750) | 183 (1262) |

Over the 53 convolutions of the network: Tessera 18.46 ms with default configurations, 13.28 ms with the best found; MPS 9.01 ms.
