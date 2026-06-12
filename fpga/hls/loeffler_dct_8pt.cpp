/**
 * 8-point DCT-II — Loeffler flowgraph (11 multiplications).
 * HLS target: Pynq Zynq-7000 PL.
 */
#include "loeffler_dct_8pt.h"

static const fixed_t C1 = 0.098017140330;
static const fixed_t C3 = 0.290284677254;
static const fixed_t C4 = 0.707106781187;
static const fixed_t C6 = 0.831469612173;
static const fixed_t S1 = 0.098017140330;
static const fixed_t S3 = 0.956940335732;
static const fixed_t S6 = 0.555570233020;

void loeffler_dct_8pt(fixed_t input[8], fixed_t output[8]) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE bram port=input
#pragma HLS INTERFACE bram port=output
#pragma HLS PIPELINE II=1

    fixed_t x[8];
#pragma HLS ARRAY_PARTITION variable=x complete
    for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
        x[i] = input[i];
    }

    fixed_t s07 = x[0] + x[7];
    fixed_t s12 = x[1] + x[2];
    fixed_t s34 = x[3] + x[4];
    fixed_t s56 = x[5] + x[6];
    fixed_t d07 = x[0] - x[7];
    fixed_t d12 = x[1] - x[2];
    fixed_t d34 = x[3] - x[4];
    fixed_t d56 = x[5] - x[6];
    fixed_t d1256 = d12 - d56;
    fixed_t s5634 = d56 - s34;
    fixed_t d0734 = s07 - s34;
    fixed_t dtilde1256 = d12 - s56;

    output[0] = C4 * ((s07 + s12) + (s34 + s56));
    output[4] = C4 * ((s07 + s34) - (s12 + s56));
    output[2] = C6 * d1256 - S6 * d0734;
    output[6] = S6 * d1256 + C6 * d0734;
    output[1] = C1 * (d07 + C4 * dtilde1256) + S1 * (d34 + C4 * s5634);
    output[7] = S1 * (d07 + C4 * dtilde1256) - C1 * (d34 + C4 * s5634);
    output[3] = C3 * (d07 - C4 * dtilde1256) - S3 * (d34 - C4 * s5634);
    output[5] = S3 * (d07 - C4 * dtilde1256) + C3 * (d34 - C4 * s5634);
}
