#include "sa3_orchestrator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <mlx/mlx.h>

#include "samel_common.h"   // to_row_contiguous

namespace sa3 {
namespace orch {

// ── patch_audio (free helper) ────────────────────────────────────────
// rearrange("b c (l h) -> b (c h) l", h=patch_size) — the inverse of
// patched_decode. Takes (B, C, T_audio) and returns (B, C*patch_size,
// T_audio / patch_size). Mirrors patch_audio() in optimized/mlx/scripts/sa3_mlx.py.
static mx::array patch_audio(const mx::array& audio, int patch_size = 256) {
    const auto& s = audio.shape();
    if (s.size() != 3) {
        throw std::runtime_error("patch_audio expects (B, C, T) shape");
    }
    const int B = s[0], C = s[1], T = s[2];
    if (T % patch_size != 0) {
        throw std::runtime_error("patch_audio: T=" + std::to_string(T) +
                                 " not divisible by patch_size=" + std::to_string(patch_size));
    }
    const int L = T / patch_size;
    mx::array x = mx::reshape(audio, {B, C, L, patch_size});
    x = mx::transpose(x, {0, 1, 3, 2});               // (B, C, patch, L)
    return mx::reshape(x, {B, C * patch_size, L});
}

// Encode raw init audio → init latents in dit_dtype.
//
// audio: (channels=2, samples) PLANAR fp32 at SAMPLE_RATE, in [-1, 1].
// Pads/truncates to target_samples = T_lat * SAMPLES_PER_LATENT before encoding.
// Returns (1, IO_CHANNELS=256, T_lat) at dit_dtype, ready to mix into noise or
// build inpaint conditioning from.
static mx::array encode_init_audio(
    const Pipeline& pipe,
    const InitAudio& a,
    int T_lat)
{
    if (a.channels != 2) {
        throw std::runtime_error("encode_init_audio: expected stereo (channels=2)");
    }
    const int target_samples = T_lat * SAMPLES_PER_LATENT;

    // Pad or truncate to target_samples; layout stays planar (C, T).
    std::vector<float> buf(static_cast<size_t>(a.channels) * target_samples, 0.0f);
    const int copy_samples = std::min(a.samples, target_samples);
    for (int c = 0; c < a.channels; ++c) {
        std::copy(a.data + c * a.samples,
                  a.data + c * a.samples + copy_samples,
                  buf.data() + c * target_samples);
    }
    // (channels, target_samples) → (1, channels, target_samples)
    mx::array audio(buf.data(),
                    mx::Shape{1, a.channels, target_samples},
                    mx::float32);

    // patch + encode at fp32 (SAME-L encoder is loaded as fp32).
    mx::array patches = patch_audio(audio, /*patch_size=*/256);
    mx::array init_latents = pipe.encoder(patches);
    mx::eval(init_latents);
    return mx::astype(init_latents, pipe.dit_dtype);
}

// ── Pipeline::generate ───────────────────────────────────────────────
mx::array Pipeline::generate(
    const std::string& prompt,
    float seconds,
    int   steps,
    uint64_t seed,
    float cfg,
    float apg,
    const std::string& negative_prompt,
    float sigma_max,
    std::optional<InitAudio> init_audio,
    std::optional<std::pair<float, float>> inpaint_range_seconds) const
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

    // 4. (Optional) Encode init audio → init_latents at dit_dtype.
    std::optional<mx::array> init_latents;
    if (init_audio.has_value()) {
        init_latents = encode_init_audio(*this, *init_audio, T_lat);
    }

    // 5. Pingpong schedule + initial noise (optionally mixed with init_latents).
    mx::array sigmas = build_pingpong_schedule(steps, sigma_max, /*use_logsnr_shift=*/true);
    mx::array key = mx::random::key(seed);
    mx::array pure_noise = mx::random::normal(
        {1, dit::IO_CHANNELS, T_lat}, dit_dtype, key);

    // For a2a (init_audio but no inpaint range), mix: noise = lat*(1-σ) + n*σ.
    // For inpaint, the unmasked latent is restored by paste_back at every step
    // so we start from pure noise instead.
    mx::array noise = (init_latents.has_value() && !inpaint_range_seconds.has_value())
        ? *init_latents * (1.0f - sigma_max) + pure_noise * sigma_max
        : pure_noise;
    mx::eval(noise);

    // 6. (Optional) Build inpaint local_add_cond + paste_back.
    std::optional<mx::array> local_add_cond;
    std::optional<std::pair<mx::array, mx::array>> paste_back;
    if (inpaint_range_seconds.has_value()) {
        if (!init_latents.has_value()) {
            throw std::runtime_error("inpaint_range_seconds requires init_audio");
        }
        const float start_s = inpaint_range_seconds->first;
        const float end_s   = inpaint_range_seconds->second;
        const int s0 = std::max(0, static_cast<int>(std::round(
            start_s * SAMPLE_RATE / static_cast<float>(SAMPLES_PER_LATENT))));
        const int s1 = std::min(T_lat, static_cast<int>(std::round(
            end_s   * SAMPLE_RATE / static_cast<float>(SAMPLES_PER_LATENT))));
        if (s1 <= s0) {
            throw std::runtime_error("inpaint_range_seconds rounds to empty latent span");
        }
        // mask shape (1, 1, T_lat); 1=keep, 0=inpaint.
        std::vector<float> mask_data(T_lat, 1.0f);
        for (int i = s0; i < s1; ++i) mask_data[i] = 0.0f;
        mx::array mask = mx::array(mask_data.data(), mx::Shape{1, 1, T_lat}, mx::float32);
        mx::array masked_input = mx::astype(*init_latents, mx::float32) * mask;
        // DiT's local_add_cond expects (B, T_lat, 257) at dit_dtype:
        //   concat([mask, masked_input], axis=1) → (1, 257, T_lat) → transpose
        mx::array lac = mx::concatenate({mask, masked_input}, /*axis=*/1);
        lac = mx::transpose(lac, {0, 2, 1});
        lac = mx::astype(lac, dit_dtype);
        local_add_cond = lac;
        paste_back = std::make_pair(*init_latents, mask);
        mx::eval(*local_add_cond);
    }

    // 7. model_fn — wraps DiT call, with optional batched CFG + APG.
    //    Matches optimized/mlx/scripts/sa3_mlx.py exactly.
    const auto& dit_local = dit;
    ModelFn model_fn = [&](const mx::array& x, const mx::array& t) -> mx::array {
        if (cfg == 1.0f) {
            return dit_local(x, t, cross_attn, global_cond, local_add_cond);
        }
        // Batched cond + uncond forward in one DiT call. local_add_cond is
        // duplicated along batch dim to match (cf. sa3_mlx.py model_fn).
        std::optional<mx::array> lac2;
        if (local_add_cond.has_value()) {
            lac2 = mx::concatenate({*local_add_cond, *local_add_cond}, /*axis=*/0);
        }
        mx::array x2     = mx::concatenate({x, x}, 0);
        mx::array t2     = mx::concatenate({t, t}, 0);
        mx::array cross2 = mx::concatenate({cross_attn, null_cross_attn}, 0);
        mx::array glob2  = mx::concatenate({global_cond, global_cond}, 0);
        mx::array v_batched = dit_local(x2, t2, cross2, glob2, lac2);
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

    // 8. Sample.
    mx::array latents = sample_flow_pingpong(model_fn, noise, sigmas,
                                              /*seed=*/seed + 1, paste_back);
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
    const std::string& samel_encoder_path,
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

    // SAME-L encoder (fp32, matches Python sa3_mlx.py defaults — used only
    // for init_audio paths; text-to-audio runs ignore it).
    auto enc_loaded = mx::load_safetensors(samel_encoder_path);
    auto encoder = samel::load_samel_encoder(enc_loaded.first, mx::float32);

    // SAME-L decoder (fp32, matches Python sa3_mlx.py defaults)
    auto dec_loaded = mx::load_safetensors(samel_decoder_path);
    auto decoder = samel::load_samel_decoder(dec_loaded.first, mx::float32);

    return Pipeline{
        std::move(t5),
        std::move(conditioner),
        std::move(dit),
        std::move(encoder),
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

// ── read_wav_pcm16 ───────────────────────────────────────────────────
// Minimal RIFF/WAVE reader for 16-bit PCM. Skips through chunks until it
// finds 'fmt ' + 'data'. Returns planar (channels, samples) fp32 in [-1, 1].
// Mono input is duplicated to stereo so the orchestrator's encode path
// always gets a 2-channel buffer.
static uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
static uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8)  |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

std::vector<float> read_wav_pcm16(
    const std::string& path,
    int& out_sample_rate,
    int& out_channels,
    int& out_samples)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("read_wav_pcm16: cannot open '" + path + "'");
    }
    uint8_t hdr12[12];
    if (std::fread(hdr12, 1, 12, f) != 12 ||
        std::memcmp(hdr12, "RIFF", 4) != 0 ||
        std::memcmp(hdr12 + 8, "WAVE", 4) != 0) {
        std::fclose(f);
        throw std::runtime_error("read_wav_pcm16: not a RIFF/WAVE file: " + path);
    }

    // Walk chunks until we have fmt + data.
    int sample_rate = 0, channels = 0, bits = 0;
    uint16_t fmt = 0;
    std::vector<uint8_t> data_bytes;
    bool have_fmt = false, have_data = false;
    while (!(have_fmt && have_data)) {
        uint8_t ch[8];
        if (std::fread(ch, 1, 8, f) != 8) break;
        const uint32_t sz = read_u32_le(ch + 4);
        if (std::memcmp(ch, "fmt ", 4) == 0) {
            std::vector<uint8_t> buf(sz);
            if (std::fread(buf.data(), 1, sz, f) != sz) break;
            fmt          = read_u16_le(buf.data() + 0);
            channels     = read_u16_le(buf.data() + 2);
            sample_rate  = static_cast<int>(read_u32_le(buf.data() + 4));
            bits         = read_u16_le(buf.data() + 14);
            have_fmt = true;
        } else if (std::memcmp(ch, "data", 4) == 0) {
            data_bytes.resize(sz);
            if (std::fread(data_bytes.data(), 1, sz, f) != sz) break;
            have_data = true;
        } else {
            // Skip unknown chunk (LIST/JUNK/etc.); chunks are 2-byte aligned.
            const uint32_t skip = sz + (sz & 1u);
            if (std::fseek(f, static_cast<long>(skip), SEEK_CUR) != 0) break;
        }
    }
    std::fclose(f);
    if (!have_fmt || !have_data) {
        throw std::runtime_error("read_wav_pcm16: missing fmt or data chunk: " + path);
    }
    if (fmt != 1 /* PCM */ || bits != 16) {
        throw std::runtime_error(
            "read_wav_pcm16: only 16-bit PCM supported (fmt=" +
            std::to_string(fmt) + ", bits=" + std::to_string(bits) + "): " + path);
    }
    if (channels != 1 && channels != 2) {
        throw std::runtime_error("read_wav_pcm16: only mono or stereo supported");
    }

    const int frame_bytes = channels * 2;
    const int samples = static_cast<int>(data_bytes.size()) / frame_bytes;
    const int16_t* pcm = reinterpret_cast<const int16_t*>(data_bytes.data());

    // Always return stereo (duplicate mono to L+R) so encode_init_audio gets 2ch.
    const int out_ch = 2;
    std::vector<float> out(static_cast<size_t>(out_ch) * samples);
    for (int t = 0; t < samples; ++t) {
        const float L = pcm[t * channels + 0] / 32768.0f;
        const float R = (channels == 2) ? pcm[t * channels + 1] / 32768.0f : L;
        out[0 * samples + t] = L;
        out[1 * samples + t] = R;
    }
    out_sample_rate = sample_rate;
    out_channels    = out_ch;
    out_samples     = samples;
    return out;
}

// ── Variation orchestration ──────────────────────────────────────────
// Ports of sa3_variations.py helpers — restricted to kind="melodic" since
// that's the only kind the plugin's "preserve sound" / "free variation"
// presets are tuned for.

static std::optional<std::pair<float, float>> clamp_range(
    float start, float end, float duration)
{
    start = std::max(0.0f, std::min(start, duration));
    end   = std::max(0.0f, std::min(end,   duration));
    if (end - start < 0.08f) return std::nullopt;
    return std::make_pair(start, end);
}

static std::optional<float> bar_len_seconds(
    std::optional<float> bpm, int beats_per_bar)
{
    if (!bpm.has_value() || *bpm <= 0.0f) return std::nullopt;
    return 60.0f / *bpm * static_cast<float>(beats_per_bar);
}

// prompt_for_kind(kind="melodic", ...) from sa3_variations.py.
static std::string melodic_prompt(
    const std::string& user_prompt,
    std::optional<float> bpm,
    const std::string& key,
    const std::string& steer)
{
    std::string out =
        "TrackType: Instrument, musical loop, same instrument tone, "
        "same performance style, same recording chain";
    if (bpm.has_value() && *bpm > 0.0f) {
        // Match Python "{bpm:g}" — strips trailing zeros.
        char buf[64];
        std::snprintf(buf, sizeof(buf), ", %g BPM", *bpm);
        out += buf;
    }
    if (!key.empty()) {
        out += ", in " + key;
    }
    if (!user_prompt.empty()) {
        out += ", " + user_prompt;
    }
    if (!steer.empty()) {
        out += ", " + steer;
    }
    return out;
}

// negative_prompt_for_kind(kind="melodic") from sa3_variations.py.
static std::string melodic_negative_prompt()
{
    return "poor quality, distorted, clipping, noisy, vocals, speech, silence, "
           "abrupt cutoff, completely different instrument, different timbre, "
           "different tone, different genre, unrelated melody, sudden style change";
}

std::vector<CandidateSpec> build_free_preset(
    float    /*duration_seconds*/,
    uint64_t seed_base,
    float    noise_a2a)
{
    std::vector<CandidateSpec> out;
    out.reserve(5);
    for (int i = 0; i < 5; ++i) {
        CandidateSpec s;
        s.index           = i;
        s.mode            = "a2a";
        s.seed            = seed_base + static_cast<uint64_t>(i) * 1009ull;
        s.prompt          = "";
        s.negative_prompt = "";
        s.steer           = "free variation (unconditional)";
        s.noise           = noise_a2a;
        s.cfg             = 1.0f;
        s.apg             = 1.0f;
        s.inpaint_range_seconds.reset();
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<CandidateSpec> build_app_preset(
    float                duration_seconds,
    uint64_t             seed_base,
    const std::string&   user_prompt,
    std::optional<float> bpm,
    const std::string&   key,
    int                  beats_per_bar,
    float                noise_a2a,
    float                cfg_a2a,
    float                cfg_inpaint,
    float                apg)
{
    auto bar = bar_len_seconds(bpm, beats_per_bar);
    std::optional<std::pair<float, float>> middle_bar;
    std::optional<std::pair<float, float>> first_2_bars;
    if (bar.has_value() && duration_seconds >= *bar * 4.0f) {
        // Bar-aware masks: 1 bar centered, and the first 2 bars.
        middle_bar = clamp_range(
            duration_seconds * 0.5f - *bar * 0.5f,
            duration_seconds * 0.5f + *bar * 0.5f,
            duration_seconds);
        first_2_bars = clamp_range(0.0f, *bar * 2.0f, duration_seconds);
    } else {
        // Short loop or unknown BPM — fall back to fractional masks so the
        // preset still produces 5 candidates.
        middle_bar = clamp_range(
            duration_seconds * 0.35f,
            duration_seconds * 0.65f,
            duration_seconds);
        first_2_bars = clamp_range(0.0f, duration_seconds * 0.5f, duration_seconds);
    }

    struct Slot {
        std::string                            mode;
        std::optional<std::pair<float, float>> mask;
        float                                  noise;
        std::string                            steer;
    };
    const std::vector<Slot> slots = {
        {"a2a",                  std::nullopt, noise_a2a,
            "extra passing notes and ornaments"},
        {"a2a",                  std::nullopt, noise_a2a,
            "alternate phrasing, same harmonic feel"},
        {"a2a",                  std::nullopt, noise_a2a,
            "small melodic variation with different note choices"},
        {"inpaint_middle_bar",   middle_bar,   0.85f,
            "subtly different chord progression, same key"},
        {"inpaint_first_2_bars", first_2_bars, 0.85f,
            "small reharmonization while preserving the melodic contour"},
    };

    const std::string neg = melodic_negative_prompt();
    std::vector<CandidateSpec> out;
    out.reserve(slots.size());
    for (int i = 0; i < static_cast<int>(slots.size()); ++i) {
        const auto& sl = slots[i];
        CandidateSpec s;
        s.index                 = i;
        s.mode                  = sl.mode;
        s.seed                  = seed_base + static_cast<uint64_t>(i) * 1009ull;
        s.prompt                = melodic_prompt(user_prompt, bpm, key, sl.steer);
        s.negative_prompt       = neg;
        s.steer                 = sl.steer;
        s.noise                 = sl.noise;
        s.cfg                   = (sl.mode == "a2a") ? cfg_a2a : cfg_inpaint;
        s.apg                   = apg;
        s.inpaint_range_seconds = sl.mask;
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<VariationOutput> run_variations(
    const Pipeline&                   pipe,
    const InitAudio&                  source_audio,
    const std::vector<CandidateSpec>& specs,
    float                             duration_seconds,
    int                               steps)
{
    std::vector<VariationOutput> out;
    out.reserve(specs.size());
    for (const auto& spec : specs) {
        mx::array audio = pipe.generate(
            spec.prompt,
            duration_seconds,
            steps,
            spec.seed,
            spec.cfg,
            spec.apg,
            spec.negative_prompt,
            spec.noise,
            source_audio,
            spec.inpaint_range_seconds);
        // (channels=2, samples) fp32, copy into a planar std::vector<float>
        // so callers don't need MLX in their headers.
        audio = samel::to_row_contiguous(mx::astype(audio, mx::float32));
        mx::eval(audio);
        if (audio.ndim() != 2) {
            throw std::runtime_error("run_variations: expected (channels, samples) output");
        }
        const int channels = audio.shape()[0];
        const int samples  = audio.shape()[1];
        const float* src = audio.data<float>();
        VariationOutput v;
        v.spec     = spec;
        v.channels = channels;
        v.samples  = samples;
        v.audio.assign(src, src + static_cast<size_t>(channels) * samples);
        out.push_back(std::move(v));
    }
    return out;
}

}  // namespace orch
}  // namespace sa3
