# Compatibility Matrix

This matrix records verified software envelopes, not operational fitness. A platform is supported
for release only after its dedicated CI or controlled release evidence is retained.

| Platform | Python | Core path | Optional models | Release status |
|---|---|---|---|---|
| Ubuntu 24.04 x86-64 | 3.12 | `quality` CI | `all-models` CI with LightGBM and CPU PyTorch | verified development envelope |
| macOS arm64 | 3.12 | local full suite and distribution build | locally available runtimes only | provisional; dedicated CI absent |
| Windows | 3.12 | bootstrap script documented | - | not verified |
| Other Linux distributions | 3.12 | - | - | not verified |
| Any platform | other Python versions | - | - | unsupported by the current package contract |

## Compatibility rules

- Python remains constrained to `>=3.12,<3.13` until a separately tested matrix is approved.
- Linux CI is the authoritative automated development envelope.
- macOS local success does not substitute for dedicated release CI.
- Windows bootstrap availability is not a verification result.
- Determinism is defined within a recorded software and hardware envelope; cross-platform bitwise
  identity is not claimed.
- Scale results must record platform, Python version, entity count, elapsed time, peak resident
  memory, seed, and evidence digest without including row-level data.
