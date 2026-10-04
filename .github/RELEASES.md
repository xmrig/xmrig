# XMRig release builds

Push a `v*` tag to run `.github/workflows/release.yml`. All build jobs must
succeed before archives and SHA256SUMS are published to GitHub Releases.
Manual workflow runs build downloadable Actions artifacts without publishing.

| OS | Architectures |
| --- | --- |
| Linux (glibc, Debian 13 or compatible) | x86, x64, ARMv7, ARMv8, RISC-V 64 |
| Windows | x86, x64 |
| macOS 11+ | x64, ARMv8 (Apple Silicon) |
| Android 7+ (API 24) | x86, x64, ARMv7, ARMv8 |

These are CPU miner executables with libuv support. TLS is disabled for the
portable cross-builds because the target SDKs provide different TLS stacks.
GPU backends, hwloc, MSR tuning and DMI are disabled for portability.
Linux still requires the target system's libc. ARMv7 requires NEON.
32-bit binaries have limited address space; use a 64-bit miner for RandomX
fast mode, and use light mode where a full dataset will not fit.

Android archives contain native command-line executables, not APKs. Install
into an executable app-owned directory or use adb to copy to /data/local/tmp,
then chmod +x xmrig and run ./xmrig --config config.json. Shared storage such
as /sdcard does not generally allow executable files.

macOS archives are unsigned command-line binaries. Windows ARM64, macOS
32-bit, and Android RISC-V are not included: this workflow uses the supported
XMRig/SDK combinations above. RISC-V targets the Linux RV64GC baseline.

Every archive contains the repository's example config.json. Edit its pool
and wallet settings before mining. No operator credentials are packaged.

Reproduce with `bash scripts/release-build.sh OS ARCH`, using the compilers
listed in release.yml; Android also requires ANDROID_NDK_HOME. Dependencies
use the versions pinned by the existing scripts/build.uv.sh and
scripts/build.openssl3.sh. Builds go into build-release/ and dist/.
