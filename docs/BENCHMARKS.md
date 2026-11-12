# Benchmarks

**5504×3096 → 1920×1080 on an RTX 5090.** Both inputs represent a 10-second clip with 240 frames at 24 fps and a mono audio track. The node ran with NVIDIA NVENC P5 and `quality=14`.

| Input | Resize and encode | What the test exercises |
| --- | ---: | --- |
| Saved HEVC file with AAC | **1.00 s** median | NVDEC → FFmpeg CUDA Lanczos → NVENC. AAC packets were copied unchanged. |
| CPU-resident float32 frames and waveform | **3.04 s** median | Upload one frame at a time → CUDA Lanczos → NVENC. The script fails if this path calls a video serialization method before resizing. |
| Same frame input, original double-encode implementation | **31.77 s** | Encode to an in-memory video and then a temporary file → CPU Lanczos → NVENC. See comparison conditions below. |

The first two rows are medians of three runs after the relevant CUDA kernel had been compiled. The legacy comparison ran once on the same 240-frame tensor input with the same output size, CQ and NVENC preset. The current node took **3.025 s** in that paired run, about **10.5× faster** in this test. The legacy output kept the frame count and reported 24.10022 fps on average; the current output reported 24 fps. The legacy test measures processing time, and the filter check below separately compares the resized pixels.

The file fixture is a moving FFmpeg test pattern encoded to HEVC. The tensor fixture repeats one decoded RGB frame using a view. Every frame goes through the resize and encoding loop, while the fixture uses one frame of source storage. This measures the node with CPU-resident input and excludes the cost and memory of upstream generation or VSR. Real footage, GPU load, storage and driver versions can change the result.

The float32 frame test peaked at **270.2 MiB of additional PyTorch CUDA allocation**. This measures the node's PyTorch allocations above the starting point. FFmpeg and the driver use separate memory. The file route uses FFmpeg hardware surfaces outside the PyTorch allocator.

### Resampling check

The benchmark feeds the same 5504×3096 RGB frame to the CUDA Lanczos filter and an antialiased bilinear filter. It compares both outputs to FFmpeg's software Lanczos at 1920×1080, before video compression. Root mean square error (RMSE) is in 8-bit RGB levels. A lower value means a closer match to this specific reference.

| Filter | RGB RMSE vs. software Lanczos |
| --- | ---: |
| CUDA Lanczos | **0.1044** |
| CUDA bilinear | 0.6150 |

### Validation

- Both GPU paths delivered the expected dimensions, frame count and frame rate with audio present. The test checks that the file route copies the original AAC packets unchanged.
- Saving the file result with ComfyUI's `auto` format and codec took **0.040 s**. Audio and video packet hashes matched the resized file, which confirms the streams were remuxed in this test.
- The package was also loaded in an isolated ComfyUI. The example file graph and a `Create Video (codec=none)` graph both ran through `Save Video`.
- The tested profile is ordinary SDR delivery. HDR and uncommon codecs or color formats need separate validation.

### Test environment and reproduction

Windows, NVIDIA GeForce RTX 5090 (driver 621.01), ComfyUI 0.44.0, Python 3.13.11, PyTorch 2.10.0+cu130, Triton 3.6.0 via Triton-Windows 3.6.0.post26, PyAV 18.1.0, imageio-ffmpeg 0.6.0, and FFmpeg 7.1. [Machine-readable results](benchmarks-rtx5090.json)

Run from the repository with the same Python used for ComfyUI. The benchmark needs CUDA-enabled PyTorch, Triton and FFmpeg with `scale_cuda`, `h264_nvenc` and `hevc_nvenc`. The HEVC encoder is used to generate the high-resolution input fixture.

```sh
python tools/benchmark.py --comfyui /path/to/ComfyUI --legacy-frames 240 --json benchmark-results.json
```

The fixture is generated on the machine and stored in a temporary directory. Omit `--legacy-frames 240` to measure just the current routes.
