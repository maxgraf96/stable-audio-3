#include "sa3_orchestrator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <mlx/mlx.h>

namespace sa3 {
namespace orch {

// ── Pipeline::generate ───────────────────────────────────────────────
mx::array Pipeline::generate(
    const std::string& prompt,
    float seconds,
    int   steps,
    uint64_t seed,
    float cfg,
    float apg,
    const std::string& negative_prompt,
    float sigma_max) const
{
    // 1. T_lat from seconds. SAME-L has no even-T_lat constraint.
    const int T_lat = std::max(1, static_cast<int>(
        std::ceil(seconds * static_cast<float>(SAMPLE_RATE) /
                  static_cast<float>(SAMPLES_PER_LATENT))));

    // 2. T5Gemma encode → cross_attn (1, 257, 768) + global_cond (1, 768).
    auto enc = t5.encode({prompt}, /*max_len=*/256);
    mx::array embeds = mx::astype(enc.first,  dit_dtype);
    mx::array mask   = enc.second;
    mx::array embeds_padded = apply_prompt_padding(
        embeds, mask, mx::astype(conditioner.padding_embedding, dit_dtype));
    mx::array seconds_embed = mx::astype(conditioner.seconds_embedder(seconds), dit_dtype);  // (1, 1, 768)
    mx::array cross_attn = mx::concatenate({embeds_padded, seconds_embed}, /*axis=*/1);       // (1, 257, 768)
    // global_cond = seconds_embed[:, 0, :]  →  (1, 768).
    mx::array global_cond = mx::reshape(seconds_embed, {seconds_embed.shape()[0], dit::COND_TOKEN_DIM});

    // 3. CFG branch: build null_cross_attn for the uncond pass.
    mx::array null_cross_attn = mx::zeros_like(cross_attn);  // default zeros
    if (cfg != 1.0f && !negative_prompt.empty()) {
        auto neg_enc = t5.encode({negative_prompt}, /*max_len=*/256);
        mx::array neg_embeds = mx::astype(neg_enc.first, dit_dtype);
        mx::array neg_padded = apply_prompt_padding(
            neg_embeds, neg_enc.second,
            mx::astype(conditioner.padding_embedding, dit_dtype));
        null_cross_attn = mx::concatenate({neg_padded, seconds_embed}, /*axis=*/1);
    }
    mx::eval(cross_attn, global_cond, null_cross_attn);

    // 4. Pingpong schedule (sigma_max → 0) + initial noise.
    mx::array sigmas = build_pingpong_schedule(steps, sigma_max, /*use_logsnr_shift=*/true);
    mx::array key = mx::random::key(seed);
    mx::array noise = mx::random::normal(
        {1, dit::IO_CHANNELS, T_lat}, dit_dtype, key);
    mx::eval(noise);

    // 5. model_fn — wraps DiT call, with optional batched CFG + APG.
    //    Matches optimized/mlx/scripts/sa3_mlx.py exactly.
    const auto& dit_local = dit;
    ModelFn model_fn = [&](const mx::array& x, const mx::array& t) -> mx::array {
        if (cfg == 1.0f) {
            return dit_local(x, t, cross_attn, global_cond, std::nullopt);
        }
        // Batched cond + uncond forward in one DiT call.
        mx::array x2     = mx::concatenate({x, x}, 0);
        mx::array t2     = mx::concatenate({t, t}, 0);
        mx::array cross2 = mx::concatenate({cross_attn, null_cross_attn}, 0);
        mx::array glob2  = mx::concatenate({global_cond, global_cond}, 0);
        mx::array v_batched = dit_local(x2, t2, cross2, glob2, std::nullopt);
        auto parts = mx::split(v_batched, 2, /*axis=*/0);
        mx::array cond_v   = parts[0];
        mx::array uncond_v = parts[1];

        // Convert to denoised space in fp32 (the diff/projection math is
        // catastrophic-cancellation-prone at fp16).
        mx::array sigma = mx::astype(mx::reshape(t, {-1, 1, 1}), mx::float32);
        mx::array x_fp32       = mx::astype(x, mx::float32);
        mx::array cond_d   = x_fp32 - mx::astype(cond_v,   mx::float32) * sigma;
        mx::array uncond_d = x_fp32 - mx::astype(uncond_v, mx::float32) * sigma;
        mx::array diff = cond_d - uncond_d;

        mx::array cfg_diff = diff;
        if (apg > 0.0f) {
            // APG: project diff onto direction orthogonal to cond_d, per-sample
            // over (C, T).  Avoids over-saturation at high CFG.
            mx::array norm = mx::sqrt(mx::sum(
                cond_d * cond_d, /*axes=*/std::vector<int>{-2, -1}, /*keepdims=*/true));
            mx::array unit = cond_d / mx::maximum(norm, mx::array(1e-8f));
            mx::array parallel = mx::sum(
                diff * unit, /*axes=*/std::vector<int>{-2, -1}, /*keepdims=*/true) * unit;
            mx::array diff_orth = diff - parallel;
            if (apg >= 1.0f) {
                cfg_diff = diff_orth;
            } else {
                cfg_diff = apg * diff_orth + (1.0f - apg) * diff;
            }
        }
        mx::array cfg_d = cond_d + (cfg - 1.0f) * cfg_diff;
        mx::array cfg_v = (x_fp32 - cfg_d) / sigma;
        return mx::astype(cfg_v, x.dtype());
    };

    // 6. Sample.
    mx::array latents = sample_flow_pingpong(model_fn, noise, sigmas, /*seed=*/seed + 1);
    mx::eval(latents);

    // 7. Decode at fp32 (SAME-L is loaded as fp32).
    mx::array latents_fp32 = mx::astype(latents, mx::float32);
    // chunked is safe for any size — falls through to single-shot when T_lat ≤ kernel.
    mx::array patches = samel::decode_chunked(decoder, latents_fp32,
                                              /*chunk_size=*/128, /*overlap=*/8);
    mx::eval(patches);

    // 8. Unpatch (B, 512, T_lat*16) → (B, 2, T_lat*4096).
    mx::array audio = patched_decode(patches, /*patch_size=*/256, /*channels=*/2);
    mx::eval(audio);

    // 9. Trim to exact requested length and drop batch dim.
    const int requested_samples = static_cast<int>(
        std::round(seconds * static_cast<float>(SAMPLE_RATE)));
    const int actual_samples = audio.shape().back();
    if (requested_samples < actual_samples) {
        const auto& s = audio.shape();
        mx::Shape stop(s.begin(), s.end());
        stop.back() = requested_samples;
        audio = mx::slice(audio, mx::Shape{0, 0, 0}, stop);
    }
    // (1, 2, T) → (2, T)
    audio = mx::reshape(audio, {audio.shape()[1], audio.shape()[2]});
    mx::eval(audio);
    return audio;
}

// ── load_pipeline ────────────────────────────────────────────────────
Pipeline load_pipeline(
    const std::string& t5gemma_path,
    const std::string& dit_path,
    const std::string& samel_decoder_path,
    mx::Dtype dit_dtype)
{
    // T5Gemma (fp16 by Python default; encoder + SentencePiece in one go)
    auto t5_loaded = mx::load_safetensors(t5gemma_path);
    auto t5 = t5g::load_t5gemma(t5_loaded.first, mx::float16);

    // DiT (+ conditioner baked into the same safetensors under "cond.*")
    auto dit_loaded = mx::load_safetensors(dit_path);
    auto& dit_tensors = dit_loaded.first;
    auto conditioner = load_conditioner(dit_tensors, /*prefix=*/"cond.");
    auto dit = dit::load_dit(dit_tensors, /*T_lat=*/320, dit_dtype);

    // SAME-L decoder (fp32, matches Python sa3_mlx.py defaults)
    auto dec_loaded = mx::load_safetensors(samel_decoder_path);
    auto decoder = samel::load_samel_decoder(dec_loaded.first, mx::float32);

    return Pipeline{
        std::move(t5),
        std::move(conditioner),
        std::move(dit),
        std::move(decoder),
        dit_dtype,
    };
}

// ── save_wav_pcm16 ───────────────────────────────────────────────────
//
// 16-bit PCM little-endian WAV file format (RIFF). Minimal writer — no
// extensible chunks, just enough to interop with afplay, ffmpeg, sox.
static void write_u32_le(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xff));
}
static void write_u16_le(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
}
static void write_bytes(std::vector<uint8_t>& buf, const char* p, size_t n) {
    buf.insert(buf.end(), p, p + n);
}

void save_wav_pcm16(
    const std::string& path,
    const mx::array&   audio_in,
    int                sample_rate)
{
    // Materialize to row-contig fp32 so we can read it linearly.
    mx::array audio = mx::astype(audio_in, mx::float32);
    if (audio.ndim() != 2) {
        throw std::runtime_error("save_wav_pcm16 expects (channels, samples) shape");
    }
    audio = samel::to_row_contiguous(audio);
    mx::eval(audio);
    const int channels = audio.shape()[0];
    const int samples  = audio.shape()[1];
    if (channels != 1 && channels != 2) {
        throw std::runtime_error("save_wav_pcm16: only mono or stereo supported");
    }
    const float* src = audio.data<float>();

    // Build interleaved int16 buffer, clipped to [-1, 1] then scaled.
    std::vector<int16_t> pcm(static_cast<size_t>(samples) * channels);
    for (int t = 0; t < samples; ++t) {
        for (int c = 0; c < channels; ++c) {
            float s = src[c * samples + t];
            if (!std::isfinite(s)) {
                throw std::runtime_error("save_wav_pcm16: non-finite sample");
            }
            if      (s >  1.0f) s =  1.0f;
            else if (s < -1.0f) s = -1.0f;
            pcm[t * channels + c] = static_cast<int16_t>(std::lrint(s * 32767.0f));
        }
    }

    // Build RIFF/WAVE header
    std::vector<uint8_t> hdr;
    hdr.reserve(44);
    const uint32_t data_bytes  = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
    const uint32_t byte_rate   = static_cast<uint32_t>(sample_rate) * channels * 2;
    const uint16_t block_align = static_cast<uint16_t>(channels * 2);

    write_bytes(hdr, "RIFF", 4);
    write_u32_le(hdr, 36 + data_bytes);
    write_bytes(hdr, "WAVE", 4);
    write_bytes(hdr, "fmt ", 4);
    write_u32_le(hdr, 16);                          // fmt chunk size
    write_u16_le(hdr, 1);                           // PCM
    write_u16_le(hdr, static_cast<uint16_t>(channels));
    write_u32_le(hdr, static_cast<uint32_t>(sample_rate));
    write_u32_le(hdr, byte_rate);
    write_u16_le(hdr, block_align);
    write_u16_le(hdr, 16);                          // bits per sample
    write_bytes(hdr, "data", 4);
    write_u32_le(hdr, data_bytes);

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("save_wav_pcm16: cannot open '" + path + "' for writing");
    }
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    std::fwrite(pcm.data(), sizeof(int16_t), pcm.size(), f);
    std::fclose(f);
}

}  // namespace orch
}  // namespace sa3
