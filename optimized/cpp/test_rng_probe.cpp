// Probe: does MLX C++ random match Python for normal AND split (the pingpong path)?
#include <cstdio>
#include <mlx/mlx.h>
namespace mx = mlx::core;
static void p8(const char* tag, mx::array n) {
    n = mx::astype(n, mx::float32); mx::eval(n);
    const float* d = n.data<float>();
    std::printf("%s:", tag);
    for (int i = 0; i < 8; ++i) std::printf(" %.6f", d[i]);
    std::printf("\n");
}
int main() {
    mx::array k = mx::random::key(42);
    p8("normal(k)   ", mx::random::normal({1, 256, 22}, mx::float16, k));
    // pingpong step: key, sub = split(key); noise = normal(sub)
    auto parts = mx::random::split(k, 2);
    mx::array k1 = parts.at(0), sub = parts.at(1);
    p8("split.sub   ", mx::random::normal({1, 256, 22}, mx::float16, sub));
    p8("split.k1    ", mx::random::normal({1, 256, 22}, mx::float16, k1));
    return 0;
}
