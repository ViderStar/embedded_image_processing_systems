# FPGA Design — Pynq Zynq-7000

## Target

- **Part:** xc7z020clg400-1 (Pynq-Z2)
- **Tool:** Vitis HLS 2020+ / Vivado

## IP Cores

| Core | File | Function |
|------|------|----------|
| `loeffler_dct_8pt` | `hls/loeffler_dct_8pt.cpp` | 8-point DCT-II, ap_fixed&lt;16,6&gt; |
| `ladder_round_block` | `hls/ladder_step.cpp` | Ladder rounding node |

## Build

```bash
cd fpga/hls
vitis_hls -f ../scripts/build_hls.tcl
```

C-simulation compares output with `l2l_dct/tests/golden/dct_golden.txt`.

## Block Design (planned)

```
Zynq PS ── AXI DMA ── loeffler_dct_8pt
                  └── BRAM buffers (8×8 block)
```

## Pynq Overlay

1. Copy `l2l_dct.bit` + `l2l_dct.hwh` to board
2. Open `fpga/pynq/l2l_overlay.ipynb`
3. Compare HW vs Python golden vector

## Resource Estimates (typical HLS)

| Resource | loeffler_dct_8pt |
|----------|------------------|
| LUT | ~800 |
| FF | ~600 |
| DSP | 11 |
| BRAM | 0 |

Timing target: 100 MHz (10 ns clock).
