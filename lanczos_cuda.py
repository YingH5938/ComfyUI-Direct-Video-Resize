"""Separable, antialiased Lanczos-3 for the tensor-input CUDA path."""

import math

import torch
import triton
import triton.language as tl


def _weights(source_size, output_size, device):
    scale = source_size / output_size
    filter_scale = max(1.0, scale)
    radius = 3 * filter_scale
    half = math.ceil(radius)
    centers = (torch.arange(output_size, dtype=torch.float64) + 0.5) * scale - 0.5
    first = centers.floor().to(torch.int32) - half + 1
    offsets = torch.arange(2 * half, dtype=torch.int32)
    distance = (first[:, None] + offsets - centers[:, None]) / filter_scale
    weights = torch.sinc(distance) * torch.sinc(distance / 3)
    weights = torch.where(distance.abs() < 3, weights, 0)
    weights = (weights / weights.sum(dim=1, keepdim=True)).to(torch.float32)
    return first.to(device), weights.to(device), 2 * half


@triton.jit
def _vertical(src, first, weights, dst, src_h: tl.constexpr, dst_h: tl.constexpr,
              row_size: tl.constexpr, stride_y: tl.constexpr, stride_x: tl.constexpr,
              stride_c: tl.constexpr, taps: tl.constexpr, gain: tl.constexpr,
              block: tl.constexpr):
    column = tl.program_id(0) * block + tl.arange(0, block)
    y = tl.program_id(1)
    base = tl.load(first + y)
    offset = (column // 3) * stride_x + (column % 3) * stride_c
    value = tl.full((block,), 0, tl.float32)
    for tap in tl.static_range(taps):
        src_y = tl.maximum(0, tl.minimum(src_h - 1, base + tap))
        sample = tl.load(src + src_y * stride_y + offset, column < row_size, other=0).to(tl.float32)
        value += sample * tl.load(weights + y * taps + tap)
    tl.store(dst + y * row_size + column, value * gain, column < row_size)


@triton.jit
def _horizontal(src, first, weights, dst, src_w: tl.constexpr, dst_w: tl.constexpr,
                taps: tl.constexpr, block: tl.constexpr):
    column = tl.program_id(0) * block + tl.arange(0, block)
    y = tl.program_id(1)
    x, channel = column // 3, column % 3
    valid = x < dst_w
    base = tl.load(first + x, valid, other=0)
    value = tl.full((block,), 0, tl.float32)
    for tap in tl.static_range(taps):
        src_x = tl.maximum(0, tl.minimum(src_w - 1, base + tap))
        sample = tl.load(src + (y * src_w + src_x) * 3 + channel, valid, other=0)
        value += sample * tl.load(weights + x * taps + tap, valid, other=0)
    value = tl.floor(tl.maximum(0.0, tl.minimum(255.0, value)) + 0.5)
    tl.store(dst + y * dst_w * 3 + column, value.to(tl.uint8), valid)


class LanczosResize:
    def __init__(self, src_w, src_h, width, height, device):
        self.src_w = src_w
        self.src_h = src_h
        self.width = width
        self.height = height
        self.y_first, self.y_weights, self.y_taps = _weights(src_h, height, device)
        self.x_first, self.x_weights, self.x_taps = _weights(src_w, width, device)
        self.intermediate = torch.empty((height, src_w, 3), device=device, dtype=torch.float32)
        self.result = torch.empty((height, width, 3), device=device, dtype=torch.uint8)

    def __call__(self, frame):
        block = 256
        gain = 1.0 if frame.dtype == torch.uint8 else 255.0
        _vertical[(triton.cdiv(self.src_w * 3, block), self.height)](
            frame, self.y_first, self.y_weights, self.intermediate, self.src_h, self.height,
            self.src_w * 3, *frame.stride(), self.y_taps, gain, block,
        )
        _horizontal[(triton.cdiv(self.width * 3, block), self.height)](
            self.intermediate, self.x_first, self.x_weights, self.result,
            self.src_w, self.width, self.x_taps, block,
        )
        return self.result
