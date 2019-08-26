#!/bin/bash

readonly SCRIPT_FOLDER=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
readonly PROJECT_FOLDER="${SCRIPT_FOLDER}/.."
readonly ARCHIVE_FOLDER=~/cpu_features_archives
readonly QEMU_INSTALL=${ARCHIVE_FOLDER}/qemu
readonly DEFAULT_CMAKE_ARGS=" -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON"

function extract() {
  case $1 in
    *.tar.bz2)   tar xjf "$1"    ;;
    *.tar.xz)    tar xJf "$1"    ;;
    *.tar.gz)    tar xzf "$1"    ;;
    *)
      echo "don't know how to extract '$1'..."
      exit 1
  esac
}

function unpackifnotexists() {
  mkdir -p "${ARCHIVE_FOLDER}"
  cd "${ARCHIVE_FOLDER}" || exit
  local URL=$1
  local RELATIVE_FOLDER=$2
  local DESTINATION="${ARCHIVE_FOLDER}/${RELATIVE_FOLDER}"
  if [[  ! -d "${DESTINATION}" ]] ; then
    local ARCHIVE_NAME=$(echo ${URL} | sed 's/.*\///')
    test -f "${ARCHIVE_NAME}" || wget -q "${URL}"
    extract "${ARCHIVE_NAME}"
  fi
}

function installqemuifneeded() {
  local VERSION=${QEMU_VERSION:=2.11.1}
  local ARCHES=${QEMU_ARCHES:=arm aarch64 i386 x86_64 mips mipsel}
  local TARGETS=${QEMU_TARGETS:=$(echo "$ARCHES" | sed 's#$# #;s#\([^ ]*\) #\1-linux-user #g')}

  if echo "${VERSION} ${TARGETS}" | cmp --silent ${QEMU_INSTALL}/.build -; then
    echo "qemu ${VERSION} up to date!"
    return 0
  fi

  echo "VERSION: ${VERSION}"
  echo "TARGETS: ${TARGETS}"

  rm -rf ${QEMU_INSTALL}

  # Checking for a tarball before downloading makes testing easier :-)
  local QEMU_URL="http://wiki.qemu-project.org/download/qemu-${VERSION}.tar.xz"
  local QEMU_FOLDER="qemu-${VERSION}"
  unpackifnotexists ${QEMU_URL} ${QEMU_FOLDER}
  cd ${QEMU_FOLDER} || exit

  ./configure \
    --prefix="${QEMU_INSTALL}" \
    --target-list="${TARGETS}" \
    --disable-docs \
    --disable-sdl \
    --disable-gtk \
    --disable-gnutls \
    --disable-gcrypt \
    --disable-nettle \
    --disable-curses \
    --static

  make -j4
  make install

  echo "$VERSION $TARGETS" > ${QEMU_INSTALL}/.build
}

function assert_defined(){
  local VALUE=${1}
  : "${VALUE?"${1} needs to be defined"}"
}

function integrate() {
  cd "${PROJECT_FOLDER}" || exit
  cmake -H. -B"${BUILD_DIR}" ${DEFAULT_CMAKE_ARGS} ${CMAKE_ADDITIONAL_ARGS}
  cmake --build "${BUILD_DIR}" --target all

  if [[ -n "${QEMU_ARCH}" ]]; then
    if [[ "${QEMU_ARCH}" == "DISABLED" ]]; then
      QEMU="true || "
    else
      installqemuifneeded
      QEMU="${QEMU_INSTALL}/bin/qemu-${QEMU_ARCH} ${QEMU_ARGS}"
    fi
  else
    QEMU=""
  fi
  # Run tests
  for test_binary in ${BUILD_DIR}/test/*_test; do ${QEMU} ${test_binary}; done
  # Run demo program
  ${QEMU} "${BUILD_DIR}/list_cpu_features"
}

function expand_linaro_config() {
  assert_defined TARGET
  local LINARO_ROOT_URL=https://releases.linaro.org/components/toolchain/binaries/7.2-2017.11

  local GCC_URL=${LINARO_ROOT_URL}/${TARGET}/gcc-linaro-7.2.1-2017.11-x86_64_${TARGET}.tar.xz
  local GCC_RELATIVE_FOLDER="gcc-linaro-7.2.1-2017.11-x86_64_${TARGET}"
  unpackifnotexists "${GCC_URL}" "${GCC_RELATIVE_FOLDER}"

  local SYSROOT_URL=${LINARO_ROOT_URL}/${TARGET}/sysroot-glibc-linaro-2.25-2017.11-${TARGET}.tar.xz
  local SYSROOT_RELATIVE_FOLDER=sysroot-glibc-linaro-2.25-2017.11-${TARGET}
  unpackifnotexists "${SYSROOT_URL}" "${SYSROOT_RELATIVE_FOLDER}"

  local SYSROOT_FOLDER=${ARCHIVE_FOLDER}/${SYSROOT_RELATIVE_FOLDER}
  local GCC_FOLDER=${ARCHIVE_FOLDER}/${GCC_RELATIVE_FOLDER}

  CMAKE_ADDITIONAL_ARGS+=" -DCMAKE_SYSROOT=${SYSROOT_FOLDER}"
  CMAKE_ADDITIONAL_ARGS+=" -DCMAKE_C_COMPILER=${GCC_FOLDER}/bin/${TARGET}-gcc"
  CMAKE_ADDITIONAL_ARGS+=" -DCMAKE_CXX_COMPILER=${GCC_FOLDER}/bin/${TARGET}-g++"

  CMAKE_ADDITIONAL_ARGS+=" -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER"
  CMAKE_ADDITIONAL_ARGS+=" -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY"
  CMAKE_ADDITIONAL_ARGS+=" -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY"

  QEMU_ARGS+=" -L ${SYSROOT_FOLDER}"
  QEMU_ARGS+=" -E LD_LIBRARY_PATH=/lib"
}

function expand_codescape_config() {
  assert_defined TARGET
  local FLAVOUR=${QEMU_ARCH}-r2-hard
  local DATE=2016.05-03
  local CODESCAPE_URL=http://codescape-mips-sdk.imgtec.com/components/toolchain/${DATE}/Codescape.GNU.Tools.Package.${DATE}.for.MIPS.MTI.Linux.CentOS-5.x86_64.tar.gz
  local GCC_URL=${CODESCAPE_URL}
  local GCC_RELATIVE_FOLDER=${TARGET}/${DATE}
  unpackifnotexists "${GCC_URL}" "${GCC_RELATIVE_FOLDER}"

  local SYSROOT_URL=${CODESCAPE_URL}
  local SYSROOT_FOLDER=${ARCHIVE_FOLDER}/${GCC_RELATIVE_FOLDER}/sysroot/${FLAVOUR}
  unpackifnotexists "${SYSROOT_URL}" "${SYSROOT_RELATIVE_FOLDER}"

  CMAKE_ADDITIONAL_ARGS+=" -DENABLE_MSA=1"
  CMAKE_ADDITIONAL_ARGS+=" -DMIPS_CPU=p5600"
  CMAKE_ADDITIONAL_ARGS+=" -DCMAKE_TOOLCHAIN_FILE=cmake/mips32-linux-gcc.cmake"
  CMAKE_ADDITIONAL_ARGS+=" -DCROSS=${TARGET}-"
  CMAKE_ADDITIONAL_ARGS+=" -DCMAKE_FIND_ROOT_PATH=${ARCHIVE_FOLDER}/${GCC_RELATIVE_FOLDER}"

  QEMU_ARGS+=" -L ${SYSROOT_FOLDER}"
  QEMU_ARGS+=" -E LD_LIBRARY_PATH=/lib"
  QEMU_ARGS+=" -cpu P5600"
}

function expand_environment_and_integrate() {
  assert_defined PROJECT_FOLDER
  assert_defined TARGET

  BUILD_DIR="${PROJECT_FOLDER}/cmake_build/${TARGET}"
  mkdir -p "${BUILD_DIR}"

  CMAKE_ADDITIONAL_ARGS=""
  QEMU_ARGS=""

  case ${TOOLCHAIN} in
    LINARO)    expand_linaro_config     ;;
    CODESCAPE) expand_codescape_config  ;;
    NATIVE)    QEMU_ARCH=""             ;;
    *)
              echo "Unknown toolchain '${TOOLCHAIN}'..."
              exit 1
  esac
  integrate
}

if [ "${CONTINUOUS_INTEGRATION}" = "true" ]; then
  QEMU_ARCHES=${QEMU_ARCH}
  expand_environment_and_integrate
fi
