/* XMRig
 * Copyright 2026      ercmine     <https://github.com/ercmine>
 * Copyright 2016-2024 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 */

#include "crypto/animica/AnimicaAicfDelegate.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>


namespace xmrig {


namespace {

// Where the helper binary lives by default. The `animica` PyPI package
// installs it here system-wide; per-deploy overrides come through the
// constructor / CLI flag.
constexpr const char *kDefaultRunner = "/usr/local/bin/aicf-chat-once";


// Minimal JSON-escape for the prompt body. Handles the four characters
// that must escape in a JSON string literal (`"`, `\`, control chars,
// and `\n`) without pulling in a full JSON writer dependency on this
// hot path. Anything more elaborate (Unicode normalization, BOM
// handling) is the pool's responsibility before the prompt reaches
// the worker.
std::string jsonEscape(const std::string &s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out;
}


// Naive JSON extractor for "key": "string-or-number" — enough for the
// fields we read from the runner. A real JSON parser belongs to the
// next iteration when we wire this in to the Stratum client; for the
// scaffold-and-test phase this keeps the delegate dependency-free.
std::string findField(const std::string &json, const char *key)
{
    std::string needle = std::string("\"") + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    if (pos >= json.size()) return "";
    if (json[pos] == '"') {
        ++pos;
        auto end = pos;
        while (end < json.size() && json[end] != '"') {
            if (json[end] == '\\') end += 2; else ++end;
        }
        return json.substr(pos, end - pos);
    }
    auto end = pos;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != ' ') ++end;
    return json.substr(pos, end - pos);
}


} // anonymous namespace


AnimicaAicfDelegate::AnimicaAicfDelegate() :
    m_runnerPath(kDefaultRunner)
{
    if (const char *env = std::getenv("ANIMICA_AICF_RUNNER")) {
        if (*env) {
            m_runnerPath = env;
        }
    }
}


AnimicaAicfDelegate::AnimicaAicfDelegate(const std::string &runner_path) :
    m_runnerPath(runner_path.empty() ? kDefaultRunner : runner_path)
{
}


bool AnimicaAicfDelegate::runnerAvailable() const
{
    return ::access(m_runnerPath.c_str(), X_OK) == 0;
}


AnimicaAicfDelegate::Result AnimicaAicfDelegate::run(
    const std::string &prompt,
    const std::string &tier,
    uint32_t max_output_tokens,
    double temperature,
    std::chrono::milliseconds timeout) const
{
    Result r;

    if (!runnerAvailable()) {
        r.error = std::string("runner not found: ") + m_runnerPath;
        r.code  = "RUNNER_MISSING";
        return r;
    }

    // Build the one-line stdin spec.
    std::ostringstream spec;
    spec << "{\"prompt\":\""    << jsonEscape(prompt) << "\","
         << "\"tier\":\""        << tier              << "\","
         << "\"max_output_tokens\":" << max_output_tokens << ","
         << "\"temperature\":"   << temperature       << ","
         << "\"yolo\":true}";
    const std::string spec_str = spec.str();

    int in_pipe[2]  = {-1, -1};
    int out_pipe[2] = {-1, -1};
    if (::pipe(in_pipe) != 0 || ::pipe(out_pipe) != 0) {
        r.error = "pipe() failed";
        r.code  = "PIPE";
        return r;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        r.error = "fork() failed";
        r.code  = "FORK";
        return r;
    }

    if (pid == 0) {
        // Child: rewire stdio and exec the runner. Any failure here
        // exits with code 127 so the parent's WIFEXITED + WEXITSTATUS
        // can disambiguate.
        ::dup2(in_pipe[0],  STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::close(in_pipe[1]);
        ::close(out_pipe[0]);
        // Steer the LLM runner away from any GPU xmrig is using for
        // hash-share mining. Without this, the runner's torch backend
        // can grab CUDA context #0 and bring xmrig's GPU mining to a
        // halt (or vice-versa — they OOM each other). Operators who
        // want GPU inference for AICF should disable GPU mining (run
        // with -t N --no-cuda --no-opencl) and unset the env vars
        // we set here before invoking xmrig.
        ::setenv("ANIMICA_AICF_DEVICE", "cpu", 1);
        ::setenv("CUDA_VISIBLE_DEVICES", "", 1);
        ::execl(m_runnerPath.c_str(), m_runnerPath.c_str(), nullptr);
        ::_exit(127);
    }

    // Parent: write spec to the child's stdin, read its single JSON
    // line from stdout, wait for it to exit. The timeout enforcement
    // uses a coarse SIGTERM after `timeout` elapses — the next
    // iteration will replace this with a libuv-driven event loop.
    ::close(in_pipe[0]);
    ::close(out_pipe[1]);

    auto written = ::write(in_pipe[1], spec_str.data(), spec_str.size());
    ::close(in_pipe[1]);
    if (written != static_cast<ssize_t>(spec_str.size())) {
        r.error = "short write to runner stdin";
        r.code  = "WRITE";
        ::kill(pid, SIGKILL);
        ::waitpid(pid, nullptr, 0);
        ::close(out_pipe[0]);
        return r;
    }

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::string out;
    out.reserve(4096);
    std::array<char, 4096> buf;
    bool killed = false;
    while (true) {
        if (std::chrono::steady_clock::now() >= deadline) {
            ::kill(pid, SIGKILL);
            killed = true;
            break;
        }
        const ssize_t n = ::read(out_pipe[0], buf.data(), buf.size());
        if (n > 0) {
            out.append(buf.data(), static_cast<size_t>(n));
        } else {
            break;
        }
    }
    ::close(out_pipe[0]);

    int status = 0;
    ::waitpid(pid, &status, 0);
    if (killed) {
        r.error = "runner timed out";
        r.code  = "TIMEOUT";
        return r;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) == 127) {
        r.error = "runner failed to exec";
        r.code  = "EXEC";
        return r;
    }

    // Parse the last non-empty line — the runner guarantees one JSON
    // line per invocation, but a transitive Python warning may print
    // a line first (e.g. `transformers` deprecation messages).
    std::string line;
    {
        size_t end = out.size();
        while (end > 0 && (out[end - 1] == '\n' || out[end - 1] == '\r')) --end;
        size_t start = out.rfind('\n', end > 0 ? end - 1 : 0);
        if (start == std::string::npos) start = 0;
        else ++start;
        line = out.substr(start, end - start);
    }

    if (line.empty()) {
        r.error = "empty runner output";
        r.code  = "EMPTY";
        return r;
    }

    if (findField(line, "ok") == "true") {
        r.ok           = true;
        r.content      = findField(line, "content");
        r.minerId      = findField(line, "miner_id");
        const std::string cost = findField(line, "cost_animica");
        const std::string lat  = findField(line, "latency_ms");
        if (!cost.empty()) r.costAnimica = std::strtod(cost.c_str(), nullptr);
        if (!lat.empty())  r.latencyMs   = std::strtoull(lat.c_str(), nullptr, 10);
    } else {
        r.error = findField(line, "error");
        r.code  = findField(line, "code");
        if (r.error.empty()) r.error = line.substr(0, 256);
    }
    return r;
}


} // namespace xmrig
