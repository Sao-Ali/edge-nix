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

## Confirmed physical-board results

A user ran the transferred archive in an on-board terminal on 2026-10-23. The archive SHA-256 matched `4ff12a1c924964f387ab5fe93219a488076ab870dda7ee434816c94dc2a0ff54` before extraction. The board reported:

| Property | Observed value |
| --- | --- |
| Architecture | `aarch64` (`uname -m`) |
| Kernel | `6.16.7-g0dd6551ae96b` (`uname -r`) |
| OS | Debian GNU/Linux 13.1, trixie (`/etc/os-release`) |
| System glibc | 2.41 (`getconf GNU_LIBC_VERSION`) |
| Bundled glibc | 2.42 (`edge-run --check`) |
| ONNX Runtime | 1.24.4 (`edge-run --check`) |

`edge-run --check` returned `"passed":true`; MNIST inference returned ten scores under `Plus214_Output_0` with class 7 highest; `edge-validate` returned `"passed":true`. No Nix or ONNX Runtime installation was performed on the board. ONNX Runtime logged a GPU device-discovery warning about `/sys/class/drm/card0/device/vendor`; the CPU inference and validation succeeded. This result does not demonstrate GPU inference.

CPU flags, page size, available RAM and storage, system loader path, peak resident memory, and latency remain unmeasured. The remote SSH probe from the development machine still needs authentication, but the on-board terminal test above is a physical-device result.

## Provisional ARM64 Debian check

The archive built natively in an `aarch64-linux` Nix container and ran in a separate `debian:trixie-slim` ARM64 container with no Nix store, ONNX Runtime, or Python. That container reported Debian glibc 2.41 and Linux kernel `6.10.14-linuxkit`; `edge-run --check` reported the bundled glibc 2.42 and ONNX Runtime 1.24.4. Docker shares the host kernel, so this does not establish UNO Q kernel compatibility.

The current candidate contract is ARM64 Linux with kernel 6.1 or newer, a POSIX `/bin/sh`, and a Debian 13 userland. The archive invokes its bundled `ld-linux-aarch64.so.1` and ships glibc 2.42 plus the runtime libraries found in the native ELF dependency closure. The host's kernel, firmware, shell, filesystem, and normal process services remain external. The ordinary ELF `PT_INTERP` is `/lib/ld-linux-aarch64.so.1`; the included shell launcher explicitly invokes the bundled loader, so that system interpreter path is not used by the documented command.

The physical MNIST smoke test passed. The broader 2 GB resource and compatibility contract remains provisional until the remaining probes and measurements are recorded. CPU flags and page size are not yet constrained beyond generic AArch64.
