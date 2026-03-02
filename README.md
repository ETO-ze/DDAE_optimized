# DDAE for Seismic Random Noise Attenuation (Refactored)

This folder is a **refactored / engineering-friendly** version of the original notebook-based repo
`Deep-Denoising-Autoencoder-for-Seismic-Random-Noise-Attenuation`.

What you get:

- Stage-1 **supervised** training on synthetic noisy/clean pairs (MSE)
- Stage-2 **unsupervised/self-supervised** training on field data (correlation-based denoising loss)
- Robust correlation loss (epsilon + clipping) + optional MSE anchor
- Trace-wise windowing (patch extraction) + overlap-add reconstruction
- CLI scripts + YAML configs (VSCode-friendly)

See `docs/使用说明书.md` for a VSCode-first walkthrough.
