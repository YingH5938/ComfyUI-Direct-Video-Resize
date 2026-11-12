"""Reproduce the file and Create Video benchmarks in a ComfyUI environment."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import av
import imageio_ffmpeg
import torch
import torch.nn.functional as functional


SOURCE = (5504, 3096)
TARGET = (1920, 1080)
FPS = 24
SAMPLE_RATE = 48000
ROOT = Path(__file__).resolve().parents[1]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg(*args, data=None, capture=False):
    command = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)]
    return subprocess.run(command, input=data, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                          check=True).stdout


def make_fixture(path, frames):
    w, h = SOURCE
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=960x540:rate={FPS}",
           "-f", "lavfi", "-i", f"sine=frequency=660:sample_rate={SAMPLE_RATE}",
           "-t", frames / FPS, "-vf", f"format=nv12,hwupload_cuda,scale_cuda={w}:{h}:interp_algo=bilinear",
           "-c:v", "hevc_nvenc", "-preset", "p1", "-cq", "18", "-b:v", "0",
           "-c:a", "aac", "-b:a", "128k", path)


def load_node():
    name = "direct_video_resize_benchmark"
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate(path, dimensions, frames, fps):
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        assert (video.width, video.height, video.frames) == (*dimensions, frames), "Unexpected dimensions or frame count"
        actual_fps = float(video.average_rate)
        if fps is not None:
            assert abs(actual_fps - fps) < 0.001, f"Unexpected frame rate: {actual_fps}"
        assert len(container.streams.audio) == 1, "Missing audio"
        return actual_fps


def packet_digest(path, kind):
    digest = hashlib.sha256()
    with av.open(str(path)) as container:
        stream = getattr(container.streams, kind)[0]
        for packet in container.demux(stream):
            if packet.size:
                digest.update(bytes(packet))
    return digest.hexdigest()


def timed(node, video, repeats, quality, frames):
    seconds, peak_memory = [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        start = time.perf_counter()
        output, = node.resize(video, *TARGET, "NVIDIA NVENC", quality)
        seconds.append(round(time.perf_counter() - start, 3))
        peak_memory.append(round((torch.cuda.max_memory_allocated() - before) / 2**20, 1))
        path = Path(output.get_stream_source())
        validate(path, TARGET, frames, FPS)
        if len(seconds) < repeats:
            path.unlink()
    return output, {
        "runs_seconds": seconds,
        "median_seconds": round(statistics.median(seconds), 3),
        "peak_additional_pytorch_mib": max(peak_memory),
    }


def compare_filters(node_module, frame):
    """Compare raw RGB pixels before the video encoder sees them."""
    w, h = TARGET
    source = frame.numpy().tobytes()
    reference = ffmpeg("-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{SOURCE[0]}x{SOURCE[1]}",
                       "-i", "pipe:0", "-vf", f"scale={w}:{h}:flags=lanczos", "-pix_fmt", "rgb24",
                       "-f", "rawvideo", "pipe:1", data=source, capture=True)
    lanczos = next(node_module._tensor_frames(frame.unsqueeze(0), w, h, True))
    input_gpu = frame.to(device="cuda", dtype=torch.float32).div_(255).permute(2, 0, 1).unsqueeze(0)
    bilinear = functional.interpolate(input_gpu, size=(h, w), mode="bilinear", antialias=True, align_corners=False)
    bilinear = bilinear.squeeze(0).permute(1, 2, 0).mul(255).clamp_(0, 255).round_().cpu().contiguous()
    expected = torch.frombuffer(bytearray(reference), dtype=torch.uint8).float()
    measured = torch.frombuffer(bytearray(lanczos), dtype=torch.uint8).float()
    return {
        "reference": "FFmpeg swscale Lanczos, raw RGB before encoding",
        "rgb_levels": "0-255",
        "cuda_lanczos_rmse": round((measured - expected).square().mean().sqrt().item(), 4),
        "cuda_bilinear_rmse": round((bilinear.flatten() - expected).square().mean().sqrt().item(), 4),
    }


def compare_legacy(node, video, folder, frames, quality, types):
    """Reproduce the original tensor route: serialize twice, CPU resize, then NVENC P5."""
    start = time.perf_counter()
    in_memory_file = video.get_stream_source()
    intermediate = folder / "legacy-input.mkv"
    video.save_to(str(intermediate), format=types.VideoContainer.MKV, codec=types.VideoCodec.AUTO)
    del in_memory_file
    output = folder / "legacy-output.mp4"
    w, h = TARGET
    ffmpeg("-i", intermediate, "-map", "0:v:0", "-map", "0:a?",
           "-vf", f"scale={w}:{h}:flags=lanczos:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1",
           "-fps_mode", "passthrough", "-map_metadata", "0", "-c:v", "h264_nvenc", "-preset", "p5",
           "-rc", "vbr", "-cq", quality, "-b:v", "0", "-pix_fmt", "yuv420p", "-c:a", "copy",
           "-movflags", "+faststart", output)
    legacy_seconds = round(time.perf_counter() - start, 3)
    legacy_average_fps = validate(output, TARGET, frames, None)
    _, current = timed(node, video, 1, quality, frames)
    return {"frames": frames, "legacy_seconds": legacy_seconds, "current_seconds": current["median_seconds"],
            "legacy_average_fps": round(legacy_average_fps, 5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", required=True, type=Path, help="Directory containing ComfyUI main.py")
    parser.add_argument("--frames", type=int, default=240, help="Frames in the synthetic fixture")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--quality", type=int, default=14)
    parser.add_argument("--legacy-frames", type=int, default=0, help="Also measure the legacy route on a short sample")
    parser.add_argument("--json", type=Path, help="Optional output path for the benchmark report")
    args = parser.parse_args()
    if not (args.comfyui / "main.py").is_file():
        parser.error("--comfyui must point to the directory containing main.py")
    if args.frames < 1 or args.runs < 1 or not 0 <= args.quality <= 51 or not 0 <= args.legacy_frames <= args.frames:
        parser.error("Check the frame counts, runs and quality range")
    if not torch.cuda.is_available():
        parser.error("Run this benchmark using a CUDA-enabled ComfyUI Python environment")

    # ComfyUI parses sys.argv during import. The benchmark owns its CLI flags.
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(args.comfyui.resolve()))
    import folder_paths
    from comfy_api.latest import InputImpl, Types
    from comfyui_version import __version__ as comfy_version
    import triton

    node_module = load_node()
    node = node_module.DirectVideoResize()
    versions = subprocess.check_output([FFMPEG, "-version"], text=True).splitlines()[0].split()
    report = {
        "environment": {
            "os": platform.system(), "gpu": torch.cuda.get_device_name(),
            "comfyui": comfy_version, "python": platform.python_version(), "torch": torch.__version__,
            "triton": triton.__version__, "ffmpeg": versions[2],
            "pyav": av.__version__, "imageio_ffmpeg": importlib.metadata.version("imageio-ffmpeg"),
        },
        "source_wh": SOURCE, "target_wh": TARGET, "frames": args.frames, "fps": FPS, "quality": args.quality,
    }
    with tempfile.TemporaryDirectory(prefix="direct-video-resize-bench-") as directory:
        folder = Path(directory)
        folder_paths.set_temp_directory(str(folder))
        path = folder / "synthetic-source.mp4"
        make_fixture(path, args.frames)
        validate(path, SOURCE, args.frames, FPS)
        with av.open(str(path)) as container:
            raw = torch.from_numpy(next(container.decode(video=0)).to_ndarray(format="rgb24"))
        file_output, report["file"] = timed(node, InputImpl.VideoFromFile(str(path)), args.runs, args.quality, args.frames)
        audio_preserved = packet_digest(path, "audio") == packet_digest(file_output.get_stream_source(), "audio")
        assert audio_preserved, "File input audio was re-encoded"
        final = folder / "save-video-auto.mp4"
        start = time.perf_counter()
        file_output.save_to(str(final), format=Types.VideoContainer.AUTO, codec=Types.VideoCodec.AUTO)
        save_seconds = round(time.perf_counter() - start, 3)
        streams_preserved = all(packet_digest(final, kind) == packet_digest(file_output.get_stream_source(), kind)
                                for kind in ("audio", "video"))
        assert streams_preserved, "Save Video auto re-encoded the result"
        report["checks"] = {"dimensions_frame_count_fps_audio": True, "file_audio_packets_unchanged": audio_preserved,
                            "save_video_auto_packets_unchanged": streams_preserved, "save_video_auto_seconds": save_seconds}
        # One RGB frame repeated by a view. This exercises the full per-frame
        # conversion and encoding loop without allocating a 49 GB source batch.
        frame = raw.float().div_(255)

        def components(count):
            positions = torch.arange(count * SAMPLE_RATE // FPS) / SAMPLE_RATE
            waveform = (0.1 * torch.sin(positions * 2 * torch.pi * 660)).view(1, 1, -1)
            return InputImpl.VideoFromComponents(Types.VideoComponents(
                images=frame.unsqueeze(0).expand(count, -1, -1, -1),
                audio={"waveform": waveform, "sample_rate": SAMPLE_RATE}, frame_rate=FPS))

        report["filter_comparison"] = compare_filters(node_module, raw)
        video = components(args.frames)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("Frame input was unexpectedly encoded before resize")

        video.get_stream_source = forbidden
        video.save_to = forbidden
        _, report["cpu_resident_frames"] = timed(node, video, args.runs, args.quality, args.frames)
        report["checks"]["no_intermediate_video_from_components"] = True
        if args.legacy_frames:
            report["legacy_comparison"] = compare_legacy(node, components(args.legacy_frames), folder,
                                                         args.legacy_frames, args.quality, Types)
    payload = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
