# Arduino UNO Q environment record

The first inference target is the QRB2210 Linux MPU, not the STM32U585 MCU. [Arduino's board page](https://docs.arduino.cc/hardware/uno-q/) identifies Debian Linux on the MPU and a quad-core Cortex-A53. [Qualcomm's board page](https://www.qualcomm.com/developer/hardware/arduino-uno-q) lists 2 GB and 4 GB variants. The bundle is sized for the 2 GB variant, but peak memory on that board is not yet measured.

## Physical board probe

Run the following on the UNO Q, then record the output with the bundle test. These commands do not modify the board.

```sh
uname -a
uname -m
getconf LONG_BIT
cat /etc/os-release
getconf GNU_LIBC_VERSION
cat /proc/cpuinfo
cat /proc/meminfo | head -10
df -h / /tmp
ls -l /lib/ld-linux-aarch64.so.1 /lib/aarch64-linux-gnu/ld-linux-aarch64.so.1
getconf PAGESIZE
```

Physical UNO Q results: **PENDING**. The configured SSH destination `arduino@192.168.1.247` was reachable on 2026-10-22, but authentication failed (`Permission denied (publickey,password)`). No OS, glibc, kernel, CPU flags, RAM, eMMC, loader path, or latency value has been confirmed on that board. The build does not infer these properties from the board name.

## Provisional ARM64 Debian check

The archive built natively in an `aarch64-linux` Nix container and ran in a separate `debian:trixie-slim` ARM64 container with no Nix store, ONNX Runtime, or Python. That container reported Debian glibc 2.41 and Linux kernel `6.10.14-linuxkit`; `edge-run --check` reported the bundled glibc 2.42 and ONNX Runtime 1.24.4. Docker shares the host kernel, so this does not establish UNO Q kernel compatibility.

The current candidate contract is ARM64 Linux with kernel 6.1 or newer, a POSIX `/bin/sh`, and a Debian 13 userland. The archive invokes its bundled `ld-linux-aarch64.so.1` and ships glibc 2.42 plus the runtime libraries found in the native ELF dependency closure. The host's kernel, firmware, shell, filesystem, and normal process services remain external. The ordinary ELF `PT_INTERP` is `/lib/ld-linux-aarch64.so.1`; the included shell launcher explicitly invokes the bundled loader, so that system interpreter path is not used by the documented command.

This is a compatibility candidate until the physical probe and run pass. CPU flags and page size are not yet constrained beyond generic AArch64; record them before claiming board support.
