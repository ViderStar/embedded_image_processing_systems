/**
 * Ladder rounding step for 8x8 block (fixed-point Q16 simulation in HLS).
 */
#include <ap_fixed.h>

typedef ap_fixed<16, 6> fixed_t;

void ladder_round_block(fixed_t block[64], int lost_bits) {
#pragma HLS INTERFACE s_axilite port=lost_bits bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE bram port=block

    for (int i = 0; i < 64; i++) {
#pragma HLS PIPELINE II=1
        ap_int<32> v = block[i].range();
        ap_int<32> half = 1 << (lost_bits - 1);
        ap_int<32> rounded = ((v + half) >> lost_bits) << lost_bits;
        block[i] = (fixed_t)rounded;
    }
}
