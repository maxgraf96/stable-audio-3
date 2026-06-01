// Phase 4b parity gate — the C++ ring buffer must reproduce batch euler.
//
// Mirrors realtime/bench/phase1_parity.py, self-consistent within C++:
//   (A) depth-1 ring == standalone euler trajectory          (bit-exact, fp32)
//   (B) depth-D ring == B=D batched euler (same kernels)      (bit-exact, fp32)
//
// Uses real sm-music DiT weights at fp32 with deterministic dummy conditioning
// (the gate tests ring orchestration, not generation quality). Run:
//   ./test_stream_parity <dit_sm_safetensors> [T_lat] [steps] [depth]
#include <cstdio>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "dit_sm.h"
#include "samel_common.h"   // to_row_contiguous
#include "sa3_pipeline.h"   // build_pingpong_schedule
#include "stream.h"
#include "stream_ode.h"

namespace mx = mlx::core;
using namespace sa3;

static double max_abs_diff(const mx::array& a_in, const mx::array& b_in) {
    mx::array a = samel::to_row_contiguous(mx::astype(a_in, mx::float32));
    mx::array b = samel::to_row_contiguous(mx::astype(b_in, mx::float32));
    mx::eval(a, b);
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs((double)pa[i] - pb[i]));
    return m;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <dit_sm.safetensors> [T] [steps] [depth]\n", argv[0]); return 2; }
    const std::string path = argv[1];
    const int T = argc > 2 ? std::stoi(argv[2]) : 16;
    const int STEPS = argc > 3 ? std::stoi(argv[3]) : 8;
    const int DEPTH = argc > 4 ? std::stoi(argv[4]) : 4;
    const float SIGMA = 0.5f;
    const mx::Dtype DT = mx::float32;   // fp32 -> bit-exact gate (no fp16 batch-kernel noise)

    auto loaded = mx::load_safetensors(path);
    auto dit_model = dit_sm::load_dit(loaded.first, T, DT);

    // Deterministic dummy conditioning (shared by ref + ring).
    auto make = [&](std::vector<int> shape) {
        int total = 1; for (int s : shape) total *= s;
        mx::array a = mx::arange(0, total, 1, mx::float32) / (float)total;
        return mx::astype(mx::reshape(a, mx::Shape(shape.begin(), shape.end())), DT);
    };
    mx::array cross = make({1, 257, dit_sm::COND_TOKEN_DIM});
    mx::array gcond = make({1, dit_sm::GLOBAL_COND_DIM});
    // Dummy "source latent" so the a2a init mix path is exercised.
    mx::array init_latent = make({1, 256, T});

    auto dit_fwd = [&](const mx::array& x, const mx::array& t,
                       const mx::array& c, const mx::array& g) {
        return dit_model(x, t, c, g);   // local_add_cond = nullopt
    };

    // init noise for a seed, matching StreamPipeline::init_slot exactly.
    auto init_noise = [&](uint64_t seed) {
        mx::array noise = mx::random::normal({1, 256, T}, DT, mx::random::key(seed));
        return mx::astype(init_latent, DT) * mx::array(1.0f - SIGMA, DT)
             + noise * mx::array(SIGMA, DT);
    };

    // Standalone euler over one clip (mirrors sample_flow_euler).
    auto euler_ref = [&](const mx::array& x0, const mx::array& cross_b,
                         const mx::array& gcond_b) {
        mx::array sigmas = build_pingpong_schedule(STEPS, SIGMA, true);
        mx::array x = x0;
        for (int i = 0; i < STEPS; ++i) {
            mx::array tc = mx::take(sigmas, i, 0);
            mx::array tn = mx::take(sigmas, i + 1, 0);
            mx::array t = tc * mx::ones({x.shape()[0]}, DT);
            mx::array v = dit_fwd(x, t, cross_b, gcond_b);
            mx::array dt = mx::astype(mx::reshape(tn - tc, {-1, 1, 1}), DT);
            x = x + dt * v;
            mx::eval(x);
        }
        return x;
    };

    auto make_req = [&](uint64_t seed) {
        stream::SlotRequest r{cross, gcond, seed, SIGMA, init_latent};
        return r;
    };

    bool all_pass = true;

    // ── (A) depth-1 ring vs standalone euler ──
    {
        mx::array ref = euler_ref(init_noise(1528), cross, gcond);
        mx::eval(ref);
        stream::StreamPipeline pipe(dit_fwd, T, STEPS, 1, DT);
        pipe.submit(make_req(1528));
        std::optional<mx::array> eng;
        for (int i = 0; i < STEPS + 4 && !eng; ++i) eng = pipe.tick();
        double d = eng ? max_abs_diff(ref, *eng) : 1e9;
        bool ok = eng.has_value() && d == 0.0;
        all_pass &= ok;
        std::printf("(A) depth-1 ring vs euler : max|d|=%.2e  %s\n", d, ok ? "EXACT PASS" : "FAIL");
    }

    // ── (B) depth-D ring vs B=D batched euler ──
    {
        std::vector<uint64_t> seeds;
        for (int i = 0; i < DEPTH; ++i) seeds.push_back(1528 + i * 1000);
        // Batched reference: all seeds in ONE B=D euler (exact ring kernels).
        std::vector<mx::array> xs, cs, gs;
        for (uint64_t sd : seeds) { xs.push_back(init_noise(sd)); cs.push_back(cross); gs.push_back(gcond); }
        mx::array refB = euler_ref(mx::concatenate(xs, 0),
                                   mx::concatenate(cs, 0), mx::concatenate(gs, 0));
        mx::eval(refB);

        stream::StreamPipeline pipe(dit_fwd, T, STEPS, DEPTH, DT);
        for (uint64_t sd : seeds) pipe.submit(make_req(sd));
        std::vector<mx::array> outs;
        for (int i = 0; i < STEPS + DEPTH + 4 && (int)outs.size() < DEPTH; ++i) {
            auto r = pipe.tick();
            if (r) outs.push_back(*r);
        }
        bool ok = (int)outs.size() == DEPTH;
        double worst = 0;
        for (int i = 0; i < (int)outs.size(); ++i) {
            mx::array refi = mx::slice(refB, {i, 0, 0}, {i + 1, refB.shape()[1], refB.shape()[2]});
            double d = max_abs_diff(refi, outs[i]);
            worst = std::max(worst, d);
            ok &= (d == 0.0);
        }
        all_pass &= ok;
        std::printf("(B) depth-%d ring vs B=%d euler: collected %d/%d  worst max|d|=%.2e  %s\n",
                    DEPTH, DEPTH, (int)outs.size(), DEPTH, worst, ok ? "EXACT PASS" : "FAIL");
    }

    // Helper: run a single depth-1 slot to completion and return its latent.
    auto run_one = [&](const stream::SlotRequest& req) {
        stream::StreamPipeline pipe(dit_fwd, T, STEPS, 1, DT);
        pipe.submit(req);
        std::optional<mx::array> out;
        for (int i = 0; i < STEPS + 4 && !out; ++i) out = pipe.tick();
        return *out;
    };

    // Baseline euler completion for the curve/CFG sanity checks (seed 1528).
    mx::array euler_base = run_one(make_req(1528));
    mx::eval(euler_base);

    // ── (C) velocity_scale = 1.0 is a no-op (bit-exact to plain euler) ──
    {
        stream::SlotRequest r = make_req(1528);
        r.velocity_scale = stream::CurveValue{1.0f};
        double d = max_abs_diff(euler_base, run_one(r));
        bool ok = d == 0.0;
        all_pass &= ok;
        std::printf("(C) velocity_scale=1 == euler: max|d|=%.2e  %s\n", d, ok ? "EXACT PASS" : "FAIL");
    }

    // ── (D) sde_denoise_curve flat 0.0 -> output == source latent ──
    {
        stream::SlotRequest r = make_req(1528);
        r.sde_denoise_curve = stream::CurveValue{0.0f};
        double d = max_abs_diff(init_latent, run_one(r));   // sdc=0 anchors fully to source
        bool ok = d == 0.0;
        all_pass &= ok;
        std::printf("(D) sde flat 0.0 == source   : max|d|=%.2e  %s\n", d, ok ? "EXACT PASS" : "FAIL");
    }

    // ── (E) x0_target set but strength absent -> bit-exact to euler ──
    {
        stream::SlotRequest r = make_req(1528);
        r.x0_target = init_latent;          // target set...
        // ...but x0_target_strength left absent -> morph inactive
        double d = max_abs_diff(euler_base, run_one(r));
        bool ok = d == 0.0;
        all_pass &= ok;
        std::printf("(E) x0_target inactive==euler: max|d|=%.2e  %s\n", d, ok ? "EXACT PASS" : "FAIL");
    }

    // ── (F) shared sde curve 0.0 overrides a no-curve slot -> source ──
    {
        stream::StreamPipeline pipe(dit_fwd, T, STEPS, 1, DT);
        pipe.set_shared_curve("sde_denoise_curve", stream::CurveValue{0.0f});
        pipe.submit(make_req(1528));
        std::optional<mx::array> out;
        for (int i = 0; i < STEPS + 4 && !out; ++i) out = pipe.tick();
        double d = max_abs_diff(init_latent, *out);
        bool ok = d == 0.0;
        all_pass &= ok;
        std::printf("(F) shared sde=0 overrides    : max|d|=%.2e  %s\n", d, ok ? "EXACT PASS" : "FAIL");
    }

    // ── (G) cfg=1.0 is bit-exact to plain euler (guidance off) ──
    {
        stream::SlotRequest r = make_req(1528);
        r.cfg = 1.0f;   // explicit; has_cfg()==false
        double d = max_abs_diff(euler_base, run_one(r));
        bool ok = d == 0.0;
        all_pass &= ok;
        std::printf("(G) cfg=1.0 == euler         : max|d|=%.2e  %s\n", d, ok ? "EXACT PASS" : "FAIL");
    }

    std::printf("PHASE 4b STREAM PARITY GATE: %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
