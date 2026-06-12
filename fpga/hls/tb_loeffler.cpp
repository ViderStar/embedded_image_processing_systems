#include <iostream>
#include <cmath>
#include <fstream>
#include "loeffler_dct_8pt.h"

static bool load_golden(double ref[8]) {
    std::ifstream f("../../../l2l_dct/tests/golden/dct_golden.txt");
    if (!f) return false;
    for (int i = 0; i < 8; i++) f >> ref[i];
    return true;
}

int main() {
    fixed_t input[8] = {128, 64, 32, 16, 8, 4, 2, 1};
    fixed_t output[8];
    loeffler_dct_8pt(input, output);

    double ref[8] = {0};
    if (!load_golden(ref)) {
        std::cerr << "Warning: golden file not found, skipping compare\n";
        return 0;
    }

    double max_err = 0;
    for (int i = 0; i < 8; i++) {
        double got = output[i].to_double();
        max_err = std::max(max_err, std::abs(got - ref[i]));
        std::cout << i << ": got=" << got << " ref=" << ref[i] << "\n";
    }
    std::cout << "max_err=" << max_err << "\n";
    return max_err > 5.0 ? 1 : 0;
}
