"""Bounded-memory CUDA resize for ComfyUI VIDEO inputs.

File inputs stay on the NVDEC -> CUDA -> NVENC path. Tensor inputs are
resized on CUDA one frame at a time, then piped to NVENC without an
intermediate video or a second full-resolution image batch.
"""

from contextlib import ExitStack, suppress
from fractions import Fraction
from pathlib import Path
import logging
import math
import subprocess
import tempfile
import time
import uuid
import wave

import folder_paths
import imageio_ffmpeg
from comfy_api.latest import InputImpl


def _check_interrupt():
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


def _encoder_args(encoder, quality):
    if encoder == "NVIDIA NVENC":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                "-cq", str(quality), "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(quality)]


def _run_ffmpeg(command, frames=None):
    with tempfile.TemporaryFile(mode="w+b") as error_log:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE if frames is not None else subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=error_log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            if frames is not None:
                pipe_error = None
                try:
                    for frame in frames:
                        _check_interrupt()
                        try:
                            process.stdin.write(frame)
                        except BrokenPipeError as exc:
                            pipe_error = exc
                            break
                finally:
                    with suppress(BrokenPipeError):
                        process.stdin.close()
            while process.poll() is None:
                _check_interrupt()
                try:
                    process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                error_log.seek(0)
                detail = error_log.read().decode("utf-8", errors="replace")[-5000:]
                raise RuntimeError("FFmpeg video resize or encode failed.\n" + detail)
            if frames is not None and pipe_error is not None:
                raise RuntimeError("FFmpeg closed its frame input before the video was complete.") from pipe_error
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        finally:
            if process.stdin is not None and not process.stdin.closed:
                with suppress(BrokenPipeError, OSError):
                    process.stdin.close()


def _write_audio(audio, path, duration):
    import torch
    waveform = audio["waveform"]
    if waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError("Expected one audio track with shape [1, channels, samples]")
    pcm = waveform[0, :, :math.ceil(duration * int(audio["sample_rate"]))].detach().to(device="cpu", dtype=torch.float32)
    pcm = pcm.clamp(-1, 1).mul(32767).round().to(torch.int16).t().contiguous().numpy()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(waveform.shape[1])
        output.setsampwidth(2)
        output.setframerate(int(audio["sample_rate"]))
        output.writeframes(pcm.tobytes())


def _tensor_frames(images, width, height, use_cuda):
    import torch
    import torch.nn.functional as functional
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. GPU mode will not silently fall back to CPU.")
    if images.ndim != 4 or images.shape[-1] not in (3, 4) or images.shape[0] == 0:
        raise ValueError("Expected a nonempty IMAGE batch [frames, height, width, 3 or 4]")
    device = torch.device("cuda", torch.cuda.current_device()) if use_cuda else torch.device("cpu")
    src_h, src_w = images.shape[1:3]
    # Match the old node's centered fill crop without stretching faces.
    crop_w = min(src_w, max(1, round(src_h * width / height)))
    crop_h = min(src_h, max(1, round(src_w * height / width)))
    x, y = (src_w - crop_w) // 2, (src_h - crop_h) // 2
    resizing = (crop_h, crop_w) != (height, width)
    cuda_resize = None
    if use_cuda and resizing:
        from .lanczos_cuda import LanczosResize
        cuda_resize = LanczosResize(crop_w, crop_h, width, height, device)
    with torch.inference_mode():
        for source in images:
            _check_interrupt()
            frame = source[y:y + crop_h, x:x + crop_w, :3].detach()
            if cuda_resize is not None:
                packed = cuda_resize(frame.to(device=device))
                yield packed.cpu().numpy().tobytes()
                continue
            frame = frame.to(device=device, dtype=torch.float32)
            if source.dtype == torch.uint8:
                frame = frame / 255.0
            frame = frame.permute(2, 0, 1).unsqueeze(0)
            if resizing:
                frame = functional.interpolate(frame, size=(height, width), mode="bilinear",
                                               align_corners=False, antialias=True)
            frame = frame.squeeze(0).permute(1, 2, 0)
            packed = frame.mul(255).clamp_(0, 255).round_().to(torch.uint8).contiguous()
            # Only the resized byte frame crosses to RAM, never a full float video batch.
            data = packed.cpu().numpy().tobytes()
            del frame, packed
            yield data


def _resize_components(video, output, width, height, encoder, quality):
    components = video.get_components()  # References existing tensors, no encoding.
    fps = float(components.frame_rate)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video frame rate must be positive")
    with tempfile.TemporaryDirectory(prefix="cuda_resize_audio_", dir=folder_paths.get_temp_directory()) as temp:
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                   "-framerate", str(Fraction(fps).limit_denominator(100000)), "-i", "pipe:0"]
        audio = components.audio
        if audio is not None:
            audio_path = Path(temp) / "audio.wav"
            _write_audio(audio, audio_path, len(components.images) / fps)
            command += ["-i", str(audio_path)]
        command += ["-map", "0:v:0"]
        if audio is not None:
            command += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
        # RGB tensor frames use the sRGB/BT.709 convention of Create Video.
        command += ["-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p,setsar=1",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1",
                    "-fps_mode", "passthrough", *_encoder_args(encoder, quality),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        _run_ffmpeg(command, _tensor_frames(components.images, width, height, encoder == "NVIDIA NVENC"))


def _resize_file(video, output, width, height, encoder, quality):
    # This method is used only for file-backed objects, never tensor-backed VIDEO.
    with ExitStack() as stack:
        source = video.get_stream_source()
        if not isinstance(source, str):
            temp = stack.enter_context(tempfile.TemporaryDirectory(
                prefix="cuda_resize_input_", dir=folder_paths.get_temp_directory()))
            path = Path(temp) / "input.video"
            source.seek(0)
            with path.open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target, length=1024 * 1024)
            source = str(path)
        # Probe dimensions only: no frame decoding or full video materialization.
        import av
        with av.open(source) as container:
            stream = container.streams.video[0]
            src_w, src_h = stream.codec_context.width, stream.codec_context.height
            pixel_format = stream.codec_context.format.name if stream.codec_context.format else ""
        crop = getattr(video, "_VideoFromFile__crop", None)
        start_time, duration = video.get_active_trim_window()
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
        gpu = encoder == "NVIDIA NVENC"
        if gpu:
            command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        if start_time:
            command += ["-ss", str(start_time)]
        command += ["-i", source]
        if duration:
            command += ["-t", str(duration)]
        command += ["-map", "0:v:0", "-map", "0:a?", "-map_metadata", "0"]
        # Equal aspect ratio is entirely GPU resident. For cropped inputs, move
        # only one frame through the CPU crop filter, then return it to CUDA.
        same_ratio = src_w * height == src_h * width and crop is None
        if gpu and same_ratio:
            scale = f"scale_cuda={width}:{height}:interp_algo=lanczos:format=yuv420p,setsar=1"
        else:
            filters = []
            if gpu:
                # Download in the decoder's actual pixel format (including 10-bit).
                filters += ["hwdownload", "format=" + ("p010le" if
                             pixel_format in ("yuv420p10le", "p010le") else "nv12")]
            if crop is not None:
                from comfy_api.latest._util import normalize_crop_rect
                rect = normalize_crop_rect(*crop, src_w, src_h)
                if rect is not None:
                    cx, cy, cw, ch = rect
                    filters += [f"crop={cw}:{ch}:{cx}:{cy}"]
                    src_w, src_h = cw, ch
            # Crop first, then CUDA scales. Crop views do not build an image batch.
            cw = min(src_w, max(2, round(src_h * width / height))) // 2 * 2
            ch = min(src_h, max(2, round(src_w * height / width))) // 2 * 2
            if (cw, ch) != (src_w, src_h):
                filters += [f"crop={cw}:{ch}"]
            if gpu:
                filters += ["format=nv12", "hwupload_cuda",
                            f"scale_cuda={width}:{height}:interp_algo=lanczos:format=yuv420p"]
            else:
                filters += [f"scale={width}:{height}:flags=lanczos"]
            scale = ",".join(filters + ["setsar=1"])
        command += ["-vf", scale, "-fps_mode", "passthrough", *_encoder_args(encoder, quality),
                    "-c:a", "copy", "-movflags", "+faststart", str(output)]
        _run_ffmpeg(command)


class DirectVideoResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "width": ("INT", {"default": 1920, "min": 2, "max": 16384, "step": 2}),
            "height": ("INT", {"default": 1080, "min": 2, "max": 16384, "step": 2}),
            "encoder": (["NVIDIA NVENC", "CPU x264"], {"default": "NVIDIA NVENC"}),
            "quality": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
        }}

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("resized_video",)
    FUNCTION = "resize"
    CATEGORY = "video/resize"
    DESCRIPTION = ("CUDA Lanczos resize + NVENC P5 encoding. File input uses NVDEC/CUDA/NVENC; "
                   "Create Video input is resized one frame at a time on CUDA. "
                   "Center-crops to fit; no full-size intermediate encode or full-batch resize.")

    def resize(self, video, width, height, encoder, quality):
        if width < 2 or height < 2 or width % 2 or height % 2:
            raise ValueError("H.264 output needs positive even width and height")
        _check_interrupt()
        temp_dir = Path(folder_paths.get_temp_directory())
        temp_dir.mkdir(parents=True, exist_ok=True)
        output = temp_dir / f"direct_video_resize_{uuid.uuid4().hex}.mp4"
        started = time.perf_counter()
        try:
            if isinstance(video, InputImpl.VideoFromComponents):
                route = "CUDA frame streaming" if encoder == "NVIDIA NVENC" else "CPU frame streaming"
                _resize_components(video, output, width, height, encoder, quality)
            elif isinstance(video, InputImpl.VideoFromFile):
                route = "NVDEC/CUDA/NVENC" if encoder == "NVIDIA NVENC" else "CPU file resize"
                _resize_file(video, output, width, height, encoder, quality)
            else:
                raise TypeError("Connect a file VIDEO (Load Video) or Create Video with codec=none. "
                                "This VIDEO wrapper is not supported by the streaming path.")
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("FFmpeg finished without a video output")
            logging.info("[DirectVideoResize CUDA] %s -> %dx%d, %.3fs", route, width, height,
                         time.perf_counter() - started)
            return (InputImpl.VideoFromFile(str(output)),)
        except BaseException:
            output.unlink(missing_ok=True)
            raise


class DirectVideoResizePath(DirectVideoResize):
    @classmethod
    def INPUT_TYPES(cls):
        required = {"video_path": ("STRING", {"forceInput": True, "tooltip":
                    "Connect the saved_path output of a video saver."})}
        required.update({key: value for key, value in super().INPUT_TYPES()["required"].items() if key != "video"})
        return {"required": required}

    DESCRIPTION = ("Direct saved file -> NVDEC -> CUDA resize -> NVENC. "
                   "Connect saved_path, not the saver's original VIDEO passthrough.")

    def resize(self, video_path, width, height, encoder, quality):
        path = Path(video_path.strip().strip('"')).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")
        return super().resize(InputImpl.VideoFromFile(str(path)), width, height, encoder, quality)


NODE_CLASS_MAPPINGS = {"DirectVideoResize": DirectVideoResize, "DirectVideoResizePath": DirectVideoResizePath}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectVideoResize": "Video Resize (CUDA + NVENC, Streaming)",
    "DirectVideoResizePath": "CUDA 视频缩放（文件直连）",
}
