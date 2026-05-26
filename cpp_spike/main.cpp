// Spike: load SAME-L encoder weights from safetensors via mlx-c++,
// run one matmul against a deterministic input, print the result.
//
// Cross-check against cpp_spike/verify.py — same kernel (Metal fp32),
// same inputs, so the printed values must match bit-exactly.

#include <iostream>
#include <iomanip>
#include <string>

#include <mlx/mlx.h>

namespace mx = mlx::core;

static constexpr const char* WEIGHT_NAME = "mapping.weight";

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <path/to/same_l_encoder_f32.safetensors>\n";
        return 2;
    }
    const std::string path = argv[1];

    // 1. Load safetensors.  Returns {tensor_map, metadata_map}.
    auto loaded = mx::load_safetensors(path);
    auto& tensors = loaded.first;

    auto it = tensors.find(WEIGHT_NAME);
    if (it == tensors.end()) {
        std::cerr << "error: weight '" << WEIGHT_NAME << "' not found in " << path << "\n";
        return 1;
    }
    mx::array W = it->second;

    // 2. Build the exact same input verify.py uses:
    //    x = arange(0, 1024, fp32) * 0.01, reshape to (2, 512).
    mx::array x = mx::arange(0.0, 1024.0, 1.0, mx::float32);
    x = mx::multiply(x, mx::array(0.01f));
    x = mx::reshape(x, {2, 512});

    // 3. y = x @ W.T   ( (2,512) @ (512,1536) -> (2,1536) )
    mx::array y = mx::matmul(x, mx::transpose(W));
    mx::eval(y);

    // 4. Report shapes and the first 10 values of y[0,:].
    auto shape_str = [](const auto& s) {
        std::string r = "(";
        for (size_t i = 0; i < s.size(); ++i) {
            r += std::to_string(s[i]);
            if (i + 1 < s.size()) r += ", ";
        }
        r += ")";
        return r;
    };

    std::cout << "W: " << WEIGHT_NAME << "  shape=" << shape_str(W.shape()) << "\n";
    std::cout << "x: shape=" << shape_str(x.shape()) << "\n";
    std::cout << "y: shape=" << shape_str(y.shape()) << "\n\n";

    std::cout << "First 10 values of y[0, :10] at full precision:\n";
    const float* data = y.data<float>();   // unified memory on Apple Silicon
    std::cout.precision(10);
    for (int i = 0; i < 10; ++i) {
        std::cout << "  y[0," << std::setw(2) << i << "] = " << data[i] << "\n";
    }
    return 0;
}
