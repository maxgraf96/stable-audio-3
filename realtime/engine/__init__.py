"""Real-time streaming-diffusion engine for Stable Audio 3 (MLX / Apple Silicon).

A StreamDiffusion-style ring buffer of in-flight generations advanced one
denoising step per tick(), adapted from DEMON (ACE-Step) to SA3's rectified-flow
DiT and [B, 256, T] channels-first latents. See realtime/README or the project
memory for the roadmap.
"""

from .stream import StreamPipeline, SlotRequest, _Slot

__all__ = ["StreamPipeline", "SlotRequest", "_Slot"]
