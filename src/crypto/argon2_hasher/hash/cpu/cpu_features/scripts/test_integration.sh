source "$(dirname -- "$0")"/run_integration.sh

# Toolchains for little-endian, 64-bit ARMv8 for GNU/Linux systems
function set_aarch64-linux-gnu() {
  TOOLCHAIN=LINARO
  TARGET=aarch64-linux-gnu
  QEMU_ARCH=aarch64
}

# Toolchains for little-endian, hard-float, 32-bit ARMv7 (and earlier) for GNU/Linux systems 
function set_arm-linux-gnueabihf() {
  TOOLCHAIN=LINARO
  TARGET=arm-linux-gnueabihf
  QEMU_ARCH=arm
}

# Toolchains for little-endian, 32-bit ARMv8 for GNU/Linux systems
function set_armv8l-linux-gnueabihf() {
  TOOLCHAIN=LINARO
  TARGET=armv8l-linux-gnueabihf
  QEMU_ARCH=arm
}

# Toolchains for little-endian, soft-float, 32-bit ARMv7 (and earlier) for GNU/Linux systems
function set_arm-linux-gnueabi() {
  TOOLCHAIN=LINARO
  TARGET=arm-linux-gnueabi
  QEMU_ARCH=arm
}

# Toolchains for big-endian, 64-bit ARMv8 for GNU/Linux systems
function set_aarch64_be-linux-gnu() {
  TOOLCHAIN=LINARO
  TARGET=aarch64_be-linux-gnu
  QEMU_ARCH="DISABLED"
}

# Toolchains for big-endian, hard-float, 32-bit ARMv7 (and earlier) for GNU/Linux systems
function set_armeb-linux-gnueabihf() {
  TOOLCHAIN=LINARO
  TARGET=armeb-linux-gnueabihf
  QEMU_ARCH="DISABLED"
}

# Toolchains for big-endian, soft-float, 32-bit ARMv7 (and earlier) for GNU/Linux systems
function set_armeb-linux-gnueabi() {
  TOOLCHAIN=LINARO
  TARGET=armeb-linux-gnueabi
  QEMU_ARCH="DISABLED"
}


function set_mips() {
  TOOLCHAIN=CODESCAPE
  TARGET=mips-mti-linux-gnu
  QEMU_ARCH="DISABLED"
}

function set_native() {
  TOOLCHAIN=NATIVE
  TARGET=native
  QEMU_ARCH=""
}

ENVIRONMENTS="
  set_aarch64-linux-gnu
  set_arm-linux-gnueabihf
  set_armv8l-linux-gnueabihf
  set_arm-linux-gnueabi
  set_aarch64_be-linux-gnu
  set_armeb-linux-gnueabihf
  set_armeb-linux-gnueabi
  set_native
  set_mips
"

for SET_ENVIRONMENT in ${ENVIRONMENTS}; do
  ${SET_ENVIRONMENT}
  expand_environment_and_integrate
done
