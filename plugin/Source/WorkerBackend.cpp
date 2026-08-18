// Non-Apple backend: a resident Python process running the torch pipeline.
//
// The C++ MLX orchestrator that MlxBackend uses is Metal-only, so off Apple
// Silicon the plugin drives the same Python inference stack the studio uses
// (sa3_pipeline_torch.Pipeline) via plugin/scripts/sa3_worker.py. The worker is
// spawned once and kept warm — loading sa3-medium costs ~15 s, and paying that
// per click would make the app unusable — then fed newline-delimited JSON on
// stdin, replying with one JSON object per line on stdout. One `candidate`
// event per finished variation preserves the progressive slot-filling that the
// Apple path gets from run_variations' callback.
//
// juce::ChildProcess can't do this: it only reads a child's output, with no way
// to write to its stdin. So the pipes are set up by hand below. That code is
// Windows-only for now, which is also the only non-Apple target CMake builds.
#include "InferenceBackend.h"

#include <stdexcept>
#include <string>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#if JUCE_WINDOWS
 #include <windows.h>
 #include <dxgi.h>
#endif

namespace sa3plugin {

namespace {

// Mirrors MEDIUM_MIN_GB / MEDIUM_COMFORTABLE_GB in plugin/scripts/sa3_worker.py.
// SA3 Medium measured ~9.3 GB peak VRAM for an 8 s loop; 10.5 leaves enough
// margin for a compositor and a browser also holding VRAM. Deliberately not the
// Mac's 11 GB unified-memory figure, which budgets weights and working set in
// one pool — a different question from dedicated VRAM.
constexpr double kMediumMinGB         = 10.5;
constexpr double kMediumComfortableGB = 14.0;

// How long to wait for each kind of reply. Loading medium is ~15 s cold but the
// first run also pays HF cache validation, so the load budget is generous; a
// generate batch is ~6 s warm. These exist so a wedged worker surfaces as an
// error in the UI instead of a permanently spinning button.
constexpr int kLoadTimeoutMs     = 600000;   // 10 min
constexpr int kGenerateTimeoutMs = 900000;   // 15 min

// Where the worker script and the interpreter that can run it live.
//
// Two layouts have to work. An installed app has no repo at all:
//
//   <exe dir>\app\sa3_worker.py          runtime\Scripts\python.exe
//
// while a dev checkout runs straight out of the tree:
//
//   <repo>\plugin\scripts\sa3_worker.py  <repo>\.venv\Scripts\python.exe
//
// Installed wins, because a developer who has also installed the app would
// otherwise get whichever the walk-up happened to reach first. SA3_REPO and
// SA3_PYTHON override either, and are what the smoke test uses.
struct WorkerPaths {
    juce::File script;
    juce::File python;
    bool resolved() const { return script.existsAsFile() && python.existsAsFile(); }
};

juce::File venvPython(const juce::File& root, const juce::String& venvName)
{
   #if JUCE_WINDOWS
    return root.getChildFile(venvName).getChildFile("Scripts/python.exe");
   #else
    return root.getChildFile(venvName).getChildFile("bin/python");
   #endif
}

WorkerPaths resolveWorkerPaths()
{
    WorkerPaths p;

    // Installed layout, relative to the executable.
    const auto exeDir =
        juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory();
    p.script = exeDir.getChildFile("app/sa3_worker.py");
    p.python = venvPython(exeDir, "runtime");

    // Dev checkout: walk up looking for the worker itself, which is the thing
    // that actually has to exist.
    if (! p.resolved()) {
        juce::File repo;
        if (auto env = juce::SystemStats::getEnvironmentVariable("SA3_REPO", {}); env.isNotEmpty()) {
            juce::File f(env);
            if (f.isDirectory()) repo = f;
        }
        if (repo == juce::File{}) {
            auto dir = exeDir;
            for (int i = 0; i < 8 && dir.exists(); ++i) {
                if (dir.getChildFile("plugin/scripts/sa3_worker.py").existsAsFile()) { repo = dir; break; }
                dir = dir.getParentDirectory();
            }
        }
        if (repo != juce::File{}) {
            p.script = repo.getChildFile("plugin/scripts/sa3_worker.py");
            p.python = venvPython(repo, ".venv");
        }
    }

    // Explicit overrides win over whatever the search found.
    if (auto env = juce::SystemStats::getEnvironmentVariable("SA3_PYTHON", {}); env.isNotEmpty()) {
        juce::File f(env);
        if (f.existsAsFile()) p.python = f;
    }
    return p;
}

#if JUCE_WINDOWS

// Minimal bidirectional pipe to a child process: write a line, read lines back.
// Blocking by design — only the engine's worker thread talks to it, and that
// thread exists precisely so these waits don't touch the UI.
class WorkerProcess
{
public:
    ~WorkerProcess() { terminate(); }

    // Throws std::runtime_error with a user-facing message on any failure.
    void launch(const juce::File& python, const juce::File& script, const juce::File& logFile)
    {
        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;

        HANDLE childStdinRead = nullptr, childStdoutWrite = nullptr;
        if (! CreatePipe(&childStdinRead, &stdinWrite_, &sa, 0)
            || ! CreatePipe(&stdoutRead_, &childStdoutWrite, &sa, 0))
            throw std::runtime_error("could not create pipes for the inference worker");

        // Our ends must not be inherited, or the child holds a duplicate open
        // and we never see EOF when it exits.
        SetHandleInformation(stdinWrite_, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(stdoutRead_, HANDLE_FLAG_INHERIT, 0);

        // The worker logs progress and tracebacks to stderr. Send it to a file
        // rather than a pipe: an undrained pipe fills at 64 KB and blocks the
        // worker mid-load, and a GUI app has no console to inherit.
        logFile.getParentDirectory().createDirectory();
        HANDLE logHandle = CreateFileW(logFile.getFullPathName().toWideCharPointer(),
                                       FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                       &sa, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

        STARTUPINFOW si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput  = childStdinRead;
        si.hStdOutput = childStdoutWrite;
        si.hStdError  = (logHandle != INVALID_HANDLE_VALUE) ? logHandle : childStdoutWrite;

        // -u: unbuffered stdio. Without it Python block-buffers stdout when it
        // isn't a tty, so candidate events would arrive in one burst at exit —
        // defeating the whole point of streaming them.
        juce::String cmd = "\"" + python.getFullPathName() + "\" -u \""
                         + script.getFullPathName() + "\"";
        std::wstring cmdLine(cmd.toWideCharPointer());
        cmdLine.push_back(L'\0');

        PROCESS_INFORMATION pi{};
        const BOOL ok = CreateProcessW(
            nullptr, cmdLine.data(), nullptr, nullptr, TRUE,
            CREATE_NO_WINDOW, nullptr,
            script.getParentDirectory().getFullPathName().toWideCharPointer(),
            &si, &pi);

        // The child owns its ends now; ours must close or EOF never arrives.
        CloseHandle(childStdinRead);
        CloseHandle(childStdoutWrite);
        if (logHandle != INVALID_HANDLE_VALUE) CloseHandle(logHandle);

        if (! ok) {
            terminate();
            throw std::runtime_error(
                ("could not start the inference worker: " + python.getFullPathName()).toStdString());
        }
        CloseHandle(pi.hThread);
        process_ = pi.hProcess;
    }

    bool isRunning() const
    {
        if (process_ == nullptr) return false;
        return WaitForSingleObject(process_, 0) == WAIT_TIMEOUT;
    }

    void writeLine(const juce::String& line)
    {
        if (stdinWrite_ == nullptr) throw std::runtime_error("inference worker is not running");
        const auto utf8 = (line + "\n").toStdString();
        DWORD written = 0;
        if (! WriteFile(stdinWrite_, utf8.data(), (DWORD) utf8.size(), &written, nullptr)
            || written != utf8.size())
            throw std::runtime_error("lost the connection to the inference worker");
    }

    // Blocks for one complete line. Throws on worker death or timeout.
    juce::String readLine(int timeoutMs)
    {
        const auto deadline = juce::Time::getMillisecondCounter() + (juce::uint32) timeoutMs;
        for (;;) {
            if (auto nl = buffer_.find('\n'); nl != std::string::npos) {
                auto line = buffer_.substr(0, nl);
                buffer_.erase(0, nl + 1);
                if (! line.empty() && line.back() == '\r') line.pop_back();
                return juce::String::fromUTF8(line.data(), (int) line.size());
            }

            DWORD available = 0;
            if (! PeekNamedPipe(stdoutRead_, nullptr, 0, nullptr, &available, nullptr))
                throw std::runtime_error("the inference worker closed unexpectedly");

            if (available > 0) {
                char chunk[4096];
                DWORD got = 0;
                const DWORD want = juce::jmin<DWORD>(available, (DWORD) sizeof(chunk));
                if (! ReadFile(stdoutRead_, chunk, want, &got, nullptr) || got == 0)
                    throw std::runtime_error("the inference worker closed unexpectedly");
                buffer_.append(chunk, got);
                continue;
            }

            if (! isRunning())
                throw std::runtime_error(
                    "the inference worker exited unexpectedly — see worker.log");
            if (juce::Time::getMillisecondCounter() > deadline)
                throw std::runtime_error("timed out waiting for the inference worker");
            Sleep(5);
        }
    }

    void terminate()
    {
        if (stdinWrite_)  { CloseHandle(stdinWrite_);  stdinWrite_  = nullptr; }
        if (stdoutRead_)  { CloseHandle(stdoutRead_);  stdoutRead_  = nullptr; }
        if (process_) {
            if (WaitForSingleObject(process_, 2000) == WAIT_TIMEOUT)
                TerminateProcess(process_, 0);
            CloseHandle(process_);
            process_ = nullptr;
        }
        buffer_.clear();
    }

private:
    HANDLE      process_    = nullptr;
    HANDLE      stdinWrite_ = nullptr;
    HANDLE      stdoutRead_ = nullptr;
    std::string buffer_;
};

// Dedicated VRAM of the primary adapter, in GB, or 0 if it can't be read.
// Asking the worker would mean spawning Python and importing torch (~10 s)
// during the engine constructor, which would stall app startup; DXGI answers
// the same question immediately.
double dedicatedVideoMemoryGB()
{
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**) &factory)) || factory == nullptr)
        return 0.0;

    double best = 0.0;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0)
            best = juce::jmax(best, (double) desc.DedicatedVideoMemory / 1e9);
        adapter->Release();
        adapter = nullptr;
    }
    factory->Release();
    return best;
}

#endif  // JUCE_WINDOWS

// Read a WAV the worker just wrote into planar fp32.
BackendVariation loadVariationWav(const juce::File& file, const std::string& steer,
                                  const std::string& mode)
{
    juce::AudioFormatManager fmt;
    fmt.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(fmt.createReaderFor(file));
    if (reader == nullptr)
        throw std::runtime_error(("could not read variation: " + file.getFullPathName()).toStdString());

    const int samples  = (int) reader->lengthInSamples;
    const int channels = 2;
    juce::AudioBuffer<float> buf(channels, juce::jmax(1, samples));
    buf.clear();
    reader->read(&buf, 0, samples, 0, true, true);
    if (reader->numChannels == 1)
        buf.copyFrom(1, 0, buf, 0, 0, samples);

    BackendVariation out;
    out.channels = channels;
    out.samples  = samples;
    out.steer    = steer;
    out.mode     = mode;
    out.audio.resize((size_t) channels * (size_t) samples);
    for (int c = 0; c < channels; ++c)
        std::copy(buf.getReadPointer(c), buf.getReadPointer(c) + samples,
                  out.audio.data() + (size_t) c * (size_t) samples);
    return out;
}

// The worker's reader expects 44.1 kHz 16-bit PCM, which is also what ffmpeg
// hands sa3_variations. Write exactly that.
void writeSourceWav(const juce::File& file, const float* planar, int channels, int samples)
{
    file.deleteFile();
    juce::WavAudioFormat wav;
    std::unique_ptr<juce::FileOutputStream> stream(file.createOutputStream());
    if (stream == nullptr)
        throw std::runtime_error(("could not write source audio to " + file.getFullPathName()).toStdString());

    std::unique_ptr<juce::AudioFormatWriter> writer(
        wav.createWriterFor(stream.get(), (double) kSampleRate, 2, 16, {}, 0));
    if (writer == nullptr)
        throw std::runtime_error("could not encode source audio");
    stream.release();  // the writer owns it now

    juce::AudioBuffer<float> buf(2, samples);
    for (int c = 0; c < 2; ++c) {
        const int src = juce::jmin(c, channels - 1);
        buf.copyFrom(c, 0, planar + (size_t) src * (size_t) samples, samples);
    }
    writer->writeFromAudioSampleBuffer(buf, 0, samples);
    writer->flush();
}

const char* modelArgFor(ModelKind kind)
{
    switch (kind) {
        case ModelKind::SMALL_MUSIC: return "sm-music";
        case ModelKind::SMALL_SFX:   return "sm-sfx";
        case ModelKind::MEDIUM:
        default:                     return "medium";
    }
}

class WorkerBackend final : public InferenceBackend
{
public:
    MemoryInfo probeMemory() override
    {
        MemoryInfo info;
        const auto ramGB = (double) juce::SystemStats::getMemorySizeInMegabytes() / 1024.0;
       #if JUCE_WINDOWS
        const auto vramGB = dedicatedVideoMemoryGB();
       #else
        const auto vramGB = 0.0;
       #endif
        // With no usable GPU the models still run on CPU, just far slower, so
        // report system RAM and let the small models be the default.
        const auto totalGB = vramGB > 0.0 ? vramGB : ramGB;
        info.totalGB           = totalGB;
        info.workingSetGB      = totalGB;
        info.mediumSupported   = vramGB >= kMediumMinGB;
        info.mediumComfortable = vramGB >= kMediumComfortableGB;
        info.simulated         = false;
        info.defaultKind       = info.mediumSupported ? ModelKind::MEDIUM : ModelKind::SMALL_MUSIC;
        return info;
    }

    void load(ModelKind kind, const MemoryInfo&) override
    {
       #if ! JUCE_WINDOWS
        throw std::runtime_error(
            "the Python inference worker is only wired up on Windows so far "
            "(the pipe setup in WorkerBackend.cpp is Win32-specific)");
       #else
        const auto paths = resolveWorkerPaths();
        if (! paths.script.existsAsFile())
            throw std::runtime_error(
                "could not find the inference worker (app\\sa3_worker.py). The install "
                "looks incomplete — reinstall, or set SA3_REPO to a stable-audio-3 checkout.");
        if (! paths.python.existsAsFile())
            throw std::runtime_error(
                ("no Python runtime at " + paths.python.getFullPathName()
                 + ". Run scripts\\setup_runtime.ps1 from the install folder "
                   "(or `uv sync` in a dev checkout), or set SA3_PYTHON.").toStdString());

        if (proc_ == nullptr || ! proc_->isRunning()) {
            proc_ = std::make_unique<WorkerProcess>();
            proc_->launch(paths.python, paths.script, logFile());
        }

        juce::DynamicObject::Ptr req = new juce::DynamicObject();
        req->setProperty("cmd",     "load");
        req->setProperty("model",   modelArgFor(kind));
        req->setProperty("seconds", lastSeconds_);
        send(juce::var(req.get()));

        const auto ev = awaitEvent({ "ready" }, kLoadTimeoutMs);
        juce::ignoreUnused(ev);
        kind_   = kind;
        loaded_ = true;
       #endif
    }

    void unload() override
    {
        loaded_ = false;
       #if JUCE_WINDOWS
        // Tear the process down rather than just dropping the model: the worker
        // holds the pipeline for the life of the process, and freeing VRAM
        // before loading a different model is what keeps a 12 GB card from
        // holding two sets of weights at once.
        if (proc_ != nullptr) {
            if (proc_->isRunning()) {
                try {
                    juce::DynamicObject::Ptr req = new juce::DynamicObject();
                    req->setProperty("cmd", "quit");
                    send(juce::var(req.get()));
                } catch (const std::exception&) {
                    // Already gone — terminate() handles it.
                }
            }
            proc_.reset();
        }
       #endif
    }

    bool isLoaded() const override { return loaded_; }

    void runVariations(
        const float* init_planar,
        int channels,
        int samples,
        const GenerateRequest& req,
        float seconds,
        const std::function<void(const BackendVariation&)>& on_candidate) override
    {
       #if ! JUCE_WINDOWS
        juce::ignoreUnused(init_planar, channels, samples, req, seconds, on_candidate);
        throw std::runtime_error("the Python inference worker is only wired up on Windows so far");
       #else
        if (! loaded_ || proc_ == nullptr || ! proc_->isRunning())
            throw std::runtime_error("pipeline not loaded");

        // The pipeline is sized to the loop length, so a new duration makes the
        // worker rebuild it. Remember what we asked for so a later reload
        // (model switch) rebuilds at the same length.
        lastSeconds_ = seconds;

        auto dir = workDir();
        dir.createDirectory();
        auto srcWav = dir.getChildFile("source.wav");
        auto outDir = dir.getChildFile("variations");
        outDir.deleteRecursively();
        outDir.createDirectory();
        writeSourceWav(srcWav, init_planar, channels, samples);

        juce::DynamicObject::Ptr r = new juce::DynamicObject();
        r->setProperty("cmd",         "generate");
        r->setProperty("src",         srcWav.getFullPathName());
        r->setProperty("outdir",      outDir.getFullPathName());
        r->setProperty("preset",      juce::String(req.preset));
        r->setProperty("seconds",     (double) seconds);
        r->setProperty("noise",       (double) req.noise);
        r->setProperty("steps",       req.steps);
        r->setProperty("seed",        (juce::int64) req.seed);
        r->setProperty("userPrompt",  juce::String(req.user_prompt));
        r->setProperty("key",         juce::String(req.key));
        r->setProperty("beatsPerBar", req.beats_per_bar);
        r->setProperty("cfgA2a",      (double) req.cfg_a2a);
        r->setProperty("cfgInpaint",  (double) req.cfg_inpaint);
        r->setProperty("apg",         (double) req.apg);
        if (req.bpm.has_value()) r->setProperty("bpm", (double) *req.bpm);
        send(juce::var(r.get()));

        for (;;) {
            const auto ev = awaitEvent({ "candidate", "done" }, kGenerateTimeoutMs);
            if (ev["event"].toString() == "done") break;
            on_candidate(loadVariationWav(juce::File(ev["path"].toString()),
                                          ev["steer"].toString().toStdString(),
                                          ev["mode"].toString().toStdString()));
        }
       #endif
    }

private:
   #if JUCE_WINDOWS
    void send(const juce::var& obj) { proc_->writeLine(juce::JSON::toString(obj, true)); }

    // Read lines until one of `wanted` arrives. An `error` event is raised as
    // an exception carrying the worker's message, so a Python traceback lands
    // in the UI's error banner rather than disappearing into the log.
    juce::var awaitEvent(const juce::StringArray& wanted, int timeoutMs)
    {
        for (;;) {
            const auto line = proc_->readLine(timeoutMs);
            if (line.isEmpty()) continue;
            const auto parsed = juce::JSON::parse(line);
            if (! parsed.isObject()) continue;   // not protocol; ignore
            const auto event = parsed["event"].toString();
            if (event == "error")
                throw std::runtime_error(parsed["message"].toString().toStdString());
            if (event == "status") {
                // Progress, not a reply — surface it and keep waiting.
                reportStatus(parsed["message"].toString().toStdString());
                continue;
            }
            if (wanted.contains(event)) return parsed;
        }
    }
   #endif

    static juce::File appDataDir()
    {
        return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
            .getChildFile("SA3 Variations");
    }
    static juce::File logFile()  { return appDataDir().getChildFile("worker.log"); }
    static juce::File workDir()  { return appDataDir().getChildFile("work"); }

   #if JUCE_WINDOWS
    std::unique_ptr<WorkerProcess> proc_;
   #endif
    bool      loaded_      = false;
    ModelKind kind_        = ModelKind::MEDIUM;
    double    lastSeconds_ = 8.0;
};

}  // namespace

std::unique_ptr<InferenceBackend> InferenceBackend::create()
{
    return std::make_unique<WorkerBackend>();
}

}  // namespace sa3plugin
