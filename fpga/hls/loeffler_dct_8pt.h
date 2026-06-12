#ifndef LOEFFLER_DCT_8PT_H
#define LOEFFLER_DCT_8PT_H

#include <ap_fixed.h>

typedef ap_fixed<16, 6> fixed_t;

void loeffler_dct_8pt(fixed_t input[8], fixed_t output[8]);

#endif
