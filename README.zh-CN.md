# ComfyUI 视频直缩

**按指定宽高缩小或放大已有视频。** 节点接收和输出 ComfyUI 原生 `VIDEO`，NVIDIA 路线用 CUDA Lanczos 做像素重采样，再用 NVENC 编码。放大时，新像素来自已有画面的插值。需要 AI 超分的工作流在上游完成模型推理。

[English](README.md) · [测试数据与复现方法](docs/BENCHMARKS.md) · [示例工作流](workflows/resize-saved-video.json)

![视频文件和内存帧两种输入路线](docs/pipeline.svg)

### 为什么做这个节点

我做它是为了配合 [RelayVSR 4×](https://github.com/kopperx/RelayVSR) 二采。部分镜头跑出来约 **5500 像素宽**，按原尺寸看，个别纹理还是发糊、像被抹过，有些生成出来的纹理近看也不像真实细节。缩到 **2560×1440** 或 **1920×1080** 交付时，这些问题在观看尺寸下没那么显眼，主体线条也还清楚。我想在 ComfyUI 里直接存出这个尺寸，省去导出大文件再到剪辑软件里手动缩放的步骤。

当时试过的缩放接法，有的会先把大尺寸帧额外编码成临时视频，有的会一次处理整批画面。我就把它写成逐帧缩放、到目标尺寸再编码的节点。RelayVSR 是它的起因，普通视频和其他流程输出也能用。

`RelayVSR 4× → VSR Decode → Create Video（codec=none）→ 视频直缩 → Save Video`

### 能做什么

- **内存帧直接导出。** 接 `Create Video` 时逐帧在 CUDA 上做抗锯齿 Lanczos 缩放，然后送给 NVENC。第一次视频编码就在目标分辨率完成，省掉先把高分辨率帧编码成临时视频的时间。
- **已保存的视频走硬件处理。** 对兼容的同宽高比素材，FFmpeg 走 NVDEC 解码、CUDA Lanczos 缩放和 NVENC 编码。兼容 MP4 的音轨直接复制，帧时序透传。
- **缩放额外开销随单帧尺寸变化。** 节点复用工作缓冲并逐帧输出，避免额外堆出第二份整段浮点画面。上游 VSR 或解码节点已有的帧仍占用内存。
- **尽量保留源画面的线条和纹理。** NVIDIA 两条路线都使用 Lanczos 和 NVENC P5。用同一份 RGB 输入比较缩放结果，CUDA Lanczos 比双线性更接近 FFmpeg 的软件 Lanczos 参考。 [对照方法与数据](docs/BENCHMARKS.md#resampling-check)

### 安装与接线

在 `ComfyUI/custom_nodes` 下克隆仓库，用运行 ComfyUI 的 Python 安装依赖，然后重启。

```sh
git clone https://github.com/YingH5938/ComfyUI-Direct-Video-Resize.git
python -m pip install -r ComfyUI-Direct-Video-Resize/requirements.txt
```

便携版用 `python_embeded/python.exe` 代替 `python`。搜索 **`Video Resize (CUDA + NVENC, Streaming)`** 或 **`DirectVideoResize`**。

| 素材 | 接法 |
| --- | --- |
| 已保存的文件 | `Load Video` → **`Video Resize`** → `Save Video` |
| 二采或解码后的内存帧 | `Create Video`（`codec=none`）→ **`Video Resize`** → `Save Video` |
| 上游节点给出文件路径 | 路径接 **`CUDA 视频缩放（文件直连）`** → `Save Video` |

横屏交付可以填 **1920×1080** 或 **2560×1440**，选 **NVIDIA NVENC、quality 18**。想减轻压缩可以试 14，文件一般会更大。节点按目标比例居中裁剪，宽高填偶数。末端 `Save Video` 的格式和编码设为 `auto`，兼容的 MP4 会直接保存压缩流。

### 环境要求

- 本机验证环境是 Windows、ComfyUI 0.44.0、RTX 5090。GPU 模式需要带 `h264_nvenc` 的 FFmpeg，处理文件还需要 CUDA 解码和 `scale_cuda`。需要指定 FFmpeg 时，在启动 ComfyUI 前设置 `IMAGEIO_FFMPEG_EXE`。
- **内存帧做 CUDA Lanczos 还需要 Triton**。Windows 版请选与当前 PyTorch 和 Python 匹配的 [Triton-Windows](https://github.com/woct0rdho/triton-windows)。文件路线由 FFmpeg 缩放，无需 Triton。
- 主要验证的是常规 SDR 视频和 H.264 MP4 成片。文件路线复制 MP4 兼容音轨，内存帧路线把输入音频编码成 AAC。`CPU x264` 是手动可选路线，文件用 Lanczos，内存帧用带抗锯齿的双线性缩放。
- 在 RunningHub 等托管平台使用时，平台需要安装这个节点和运行依赖。

RTX 5090 实测，**5504×3096 → 1920×1080**，240 帧、24 fps、带音频。已保存文件路线中位数 **1.00 秒**，CPU 内存帧路线 **3.04 秒**。同一份内存帧和最终输出编码设置下，原先带中间编码的路线用时 **31.77 秒**，CUDA 路线是 **3.03 秒**，约快 **10.5 倍**。计时覆盖缩放和编码，完整方法见 [测试报告](docs/BENCHMARKS.md)。

项目采用 [MIT 许可证](LICENSE)。
