from __future__ import annotations

from typing import Any

import torch

import comfy.model_management
import node_helpers


CATEGORY = "audio/ACE-Step 1.5 XL"
ACE15_LATENTS_PER_SECOND = 48000.0 / 1920.0

LANGUAGES = [
    "en",
    "ja",
    "zh",
    "es",
    "de",
    "fr",
    "pt",
    "ru",
    "it",
    "nl",
    "pl",
    "tr",
    "vi",
    "cs",
    "fa",
    "id",
    "ko",
    "uk",
    "hu",
    "ar",
    "sv",
    "ro",
    "el",
]

TIMESIGNATURES = ["2", "3", "4", "6"]

KEYSCALES = [
    f"{root} {quality}"
    for quality in ["major", "minor"]
    for root in ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
]

DEFAULT_TAGS = (
    "romantic, moonlight, night city, love ballad, dreamy, sentimental, "
    "tender vocals, soft piano, gentle guitar, warm pads, slow tempo, "
    "atmospheric, intimate, emotional, cinematic"
)

DEFAULT_LYRICS = """[verse]
Silver moonlight on your face
Turns this crowded world to lace
Every shadow softens down
When your hand is what I've found

[chorus]
Stay with me beneath the moon
Let it paint our love in blue
Hold me close, don't let it end
In this light we start again"""


DEFAULT_TTS_LIKE_TAGS = (
    "Mandarin Chinese solo vocal, voice clone reference, clear lead vocal, intimate close-mic vocal, "
    "natural pronunciation, expressive singing, minimal accompaniment, soft piano bed, no instrumental intro, "
    "lyrics forward, clean vocal mix, gentle emotional delivery"
)

DEFAULT_TTS_LIKE_SCRIPT = """月色落在窗前
我把想念轻轻念给你听
风穿过安静的长街
像你的声音靠近我心里

请用温柔的声音唱出这些字
每一句都清楚
每一次停顿都像呼吸
让我听见熟悉的你"""


def _seconds_to_frames(seconds: float) -> int:
    return max(1, int(round(float(seconds) * ACE15_LATENTS_PER_SECOND)))


def _frames_to_seconds(frames: int) -> float:
    return float(frames) / ACE15_LATENTS_PER_SECOND


def _split_script_lines(script: str) -> list[str]:
    raw_lines = [line.strip() for line in str(script or "").splitlines()]
    lines = [line for line in raw_lines if line]
    if len(lines) <= 1 and lines:
        import re

        parts = re.split(r"(?<=[。！？!?；;，,])\s*", lines[0])
        lines = [part.strip() for part in parts if part.strip()]
    return lines


def _format_tts_like_lyrics(script: str, section_label: str, lines_per_section: int) -> str:
    script = str(script or "").strip()
    if not script:
        script = DEFAULT_TTS_LIKE_SCRIPT

    if any(line.strip().startswith("[") and line.strip().endswith("]") for line in script.splitlines()):
        return script

    lines = _split_script_lines(script)
    lines_per_section = max(1, int(lines_per_section))
    section_label = str(section_label or "verse").strip().strip("[]") or "verse"

    blocks = []
    for index in range(0, len(lines), lines_per_section):
        block_lines = lines[index:index + lines_per_section]
        blocks.append("[{}]\n{}".format(section_label, "\n".join(block_lines)))
    return "\n\n".join(blocks).strip()


def _copy_latent(latent: dict[str, Any], samples: torch.Tensor | None = None) -> dict[str, Any]:
    out = latent.copy()
    if samples is not None:
        out["samples"] = samples
    out["type"] = "audio"
    return out


def _zero_conditioning_without_reference(conditioning):
    out = []
    for item in conditioning:
        values = item[1].copy() if len(item) > 1 and isinstance(item[1], dict) else {}
        values.pop("reference_audio_timbre_latents", None)
        values.pop("audio_codes", None)

        pooled_output = values.get("pooled_output")
        if isinstance(pooled_output, torch.Tensor):
            values["pooled_output"] = torch.zeros_like(pooled_output)

        conditioning_lyrics = values.get("conditioning_lyrics")
        if isinstance(conditioning_lyrics, torch.Tensor):
            values["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)

        out.append([torch.zeros_like(item[0]), values])
    return out


def _latent_samples(latent: dict[str, Any]) -> torch.Tensor:
    if "samples" not in latent:
        raise ValueError("LATENT input does not contain a 'samples' tensor.")
    samples = latent["samples"]
    if not isinstance(samples, torch.Tensor):
        raise TypeError("LATENT samples must be a torch.Tensor.")
    if samples.ndim != 3:
        raise ValueError(
            f"ACE-Step 1.5 audio latents must have shape [batch, channels, frames], got {tuple(samples.shape)}."
        )
    if samples.shape[1] != 64:
        raise ValueError(
            f"ACE-Step 1.5 audio latents use 64 channels, got {samples.shape[1]}. "
            "Use the ACE 1.5 VAE/model from the workflow."
        )
    return samples


def _silence_latent_like(samples: torch.Tensor, frames: int) -> torch.Tensor:
    try:
        from comfy.ldm.ace.ace_step15 import get_silence_latent

        silence = get_silence_latent(frames, samples.device).to(device=samples.device, dtype=samples.dtype)
        if silence.shape[0] != samples.shape[0]:
            silence = silence.repeat(samples.shape[0], 1, 1)
        return silence[:, : samples.shape[1], :frames]
    except Exception:
        return torch.zeros(
            samples.shape[0],
            samples.shape[1],
            frames,
            device=samples.device,
            dtype=samples.dtype,
        )


def _mask_like(samples: torch.Tensor, latent: dict[str, Any] | None = None, mode: str = "overwrite") -> torch.Tensor:
    if latent is not None and mode != "overwrite" and "noise_mask" in latent:
        mask = latent["noise_mask"].clone().to(device=samples.device, dtype=samples.dtype)
        if mask.ndim != 3:
            mask = mask.reshape(mask.shape[0], 1, -1)
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        if mask.shape[-1] < samples.shape[-1]:
            pad = torch.zeros(
                mask.shape[0],
                1,
                samples.shape[-1] - mask.shape[-1],
                device=samples.device,
                dtype=samples.dtype,
            )
            mask = torch.cat([mask, pad], dim=-1)
        elif mask.shape[-1] > samples.shape[-1]:
            mask = mask[:, :, : samples.shape[-1]]
        if mask.shape[0] != samples.shape[0]:
            mask = mask.repeat(samples.shape[0], 1, 1)[: samples.shape[0]]
        return mask

    return torch.zeros(samples.shape[0], 1, samples.shape[-1], device=samples.device, dtype=samples.dtype)


def _apply_mask_mode(mask: torch.Tensor, start: int, end: int, value: float, mode: str) -> torch.Tensor:
    out = mask.clone()
    value = float(max(0.0, min(1.0, value)))
    if mode == "subtract":
        out[:, :, start:end] = 0.0
    elif mode == "add":
        out[:, :, start:end] = torch.maximum(
            out[:, :, start:end],
            torch.full_like(out[:, :, start:end], value),
        )
    else:
        out[:, :, start:end] = value
    return out


def _blend_reference_strength(samples: torch.Tensor, strength: float) -> torch.Tensor:
    strength = float(max(0.0, min(1.0, strength)))
    if strength >= 0.999:
        return samples
    silence = _silence_latent_like(samples, samples.shape[-1])
    return samples * strength + silence * (1.0 - strength)


def _encode_audio_with_vae(audio: dict[str, Any], vae: Any) -> dict[str, Any]:
    import torchaudio

    sample_rate = int(audio["sample_rate"])
    vae_sample_rate = int(getattr(vae, "audio_sample_rate", sample_rate))
    waveform = audio["waveform"]

    if vae_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)

    samples = vae.encode(waveform.movedim(1, -1))
    return {"samples": samples, "type": "audio", "sample_rate": vae_sample_rate}


def _crop_latent(latent: dict[str, Any], start_seconds: float, max_seconds: float) -> dict[str, Any]:
    samples = _latent_samples(latent)
    start_frame = max(0, _seconds_to_frames(start_seconds) if start_seconds > 0 else 0)
    end_frame = samples.shape[-1]
    if max_seconds > 0:
        end_frame = min(end_frame, start_frame + _seconds_to_frames(max_seconds))
    if start_frame >= samples.shape[-1]:
        raise ValueError("reference_start_seconds is beyond the end of the latent.")
    return _copy_latent(latent, samples[:, :, start_frame:end_frame])


class ACEStep15XLPromptLyrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {"multiline": True, "default": DEFAULT_TAGS}),
                "lyrics": ("STRING", {"multiline": True, "default": DEFAULT_LYRICS}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tags", "lyrics")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, tags: str, lyrics: str):
        return (tags.strip(), lyrics.strip())


class ACEStep15XLTTSLikePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {"multiline": True, "default": DEFAULT_TTS_LIKE_SCRIPT}),
                "voice_tags": ("STRING", {"multiline": True, "default": DEFAULT_TTS_LIKE_TAGS}),
                "section_label": (["verse", "chorus", "spoken", "narration"], {"default": "verse"}),
                "lines_per_section": ("INT", {"default": 4, "min": 1, "max": 16}),
                "bpm": ("INT", {"default": 80, "min": 10, "max": 300}),
                "duration": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 2000.0, "step": 0.1}),
                "timesignature": (TIMESIGNATURES, {"default": "4"}),
                "language": (LANGUAGES, {"default": "zh"}),
                "keyscale": (KEYSCALES, {"default": "G minor"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT", TIMESIGNATURES, LANGUAGES, KEYSCALES)
    RETURN_NAMES = ("tags", "lyrics", "bpm", "duration", "timesignature", "language", "keyscale")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        script: str,
        voice_tags: str,
        section_label: str,
        lines_per_section: int,
        bpm: int,
        duration: float,
        timesignature: str,
        language: str,
        keyscale: str,
    ):
        tags = str(voice_tags or DEFAULT_TTS_LIKE_TAGS).strip()
        lyrics = _format_tts_like_lyrics(script, section_label, lines_per_section)
        return (tags, lyrics, int(bpm), float(duration), str(timesignature), str(language), str(keyscale))


class ACEStep15XLTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "tags": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": DEFAULT_TAGS}),
                "lyrics": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": DEFAULT_LYRICS}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "bpm": ("INT", {"default": 120, "min": 10, "max": 300}),
                "duration": ("FLOAT", {"default": 120.0, "min": 0.0, "max": 2000.0, "step": 0.1}),
                "timesignature": (TIMESIGNATURES, {"default": "4"}),
                "language": (LANGUAGES, {"default": "en"}),
                "keyscale": (KEYSCALES, {"default": "E minor"}),
                "generate_audio_codes": ("BOOLEAN", {"default": True}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "tags", "lyrics")
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(
        self,
        clip,
        tags: str,
        lyrics: str,
        seed: int,
        bpm: int,
        duration: float,
        timesignature: str,
        language: str,
        keyscale: str,
        generate_audio_codes: bool,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
    ):
        tokens = clip.tokenize(
            tags,
            lyrics=lyrics,
            seed=seed,
            bpm=bpm,
            duration=duration,
            timesignature=int(timesignature),
            language=language,
            keyscale=keyscale,
            generate_audio_codes=generate_audio_codes,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return (conditioning, tags.strip(), lyrics.strip())


class ACEStep15XLEmptyLatentAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 1000.0, "step": 0.01}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "INT")
    RETURN_NAMES = ("latent", "seconds", "latent_frames")
    FUNCTION = "create"
    CATEGORY = CATEGORY

    def create(self, seconds: float, batch_size: int):
        frames = _seconds_to_frames(seconds)
        latent = torch.zeros(
            [batch_size, 64, frames],
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        return ({"samples": latent, "type": "audio"}, _frames_to_seconds(frames), frames)


class ACEStep15XLAudioToLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "INT")
    RETURN_NAMES = ("latent", "seconds", "latent_frames")
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, audio, vae):
        latent = _encode_audio_with_vae(audio, vae)
        samples = _latent_samples(latent)
        return (latent, _frames_to_seconds(samples.shape[-1]), samples.shape[-1])


class ACEStep15XLReferenceLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "reference_latent": ("LATENT",),
                "reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "reference_max_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "CONDITIONING")
    RETURN_NAMES = ("conditioning", "reference_latent", "negative_conditioning")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(
        self,
        conditioning,
        reference_latent,
        reference_strength: float,
        reference_start_seconds: float,
        reference_max_seconds: float,
    ):
        negative_conditioning = _zero_conditioning_without_reference(conditioning)
        reference_latent = _crop_latent(reference_latent, reference_start_seconds, reference_max_seconds)
        samples = _blend_reference_strength(_latent_samples(reference_latent), reference_strength)
        reference_latent = _copy_latent(reference_latent, samples)
        conditioning = node_helpers.conditioning_set_values(
            conditioning,
            {"reference_audio_timbre_latents": [samples]},
            append=True,
        )
        return (conditioning, reference_latent, negative_conditioning)


class ACEStep15XLReferenceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "reference_audio": ("AUDIO",),
                "vae": ("VAE",),
                "reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "reference_max_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "CONDITIONING")
    RETURN_NAMES = ("conditioning", "reference_latent", "negative_conditioning")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(
        self,
        conditioning,
        reference_audio,
        vae,
        reference_strength: float,
        reference_start_seconds: float,
        reference_max_seconds: float,
    ):
        latent = _encode_audio_with_vae(reference_audio, vae)
        return ACEStep15XLReferenceLatent().apply(
            conditioning,
            latent,
            reference_strength,
            reference_start_seconds,
            reference_max_seconds,
        )


class ACEStep15XLExtendLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "left_extend_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "right_extend_seconds": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "existing_mask_mode": (["overwrite", "add", "subtract"], {"default": "overwrite"}),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "STRING")
    RETURN_NAMES = ("latent", "seconds", "summary")
    FUNCTION = "extend"
    CATEGORY = CATEGORY

    def extend(self, latent, left_extend_seconds: float, right_extend_seconds: float, existing_mask_mode: str):
        samples = _latent_samples(latent)
        left_frames = 0 if left_extend_seconds <= 0 else _seconds_to_frames(left_extend_seconds)
        right_frames = 0 if right_extend_seconds <= 0 else _seconds_to_frames(right_extend_seconds)
        if left_frames == 0 and right_frames == 0:
            return (latent, _frames_to_seconds(samples.shape[-1]), "No extension requested.")

        parts = []
        if left_frames:
            parts.append(_silence_latent_like(samples, left_frames))
        parts.append(samples)
        if right_frames:
            parts.append(_silence_latent_like(samples, right_frames))
        extended = torch.cat(parts, dim=-1)

        old_mask = _mask_like(samples, latent, existing_mask_mode)
        new_mask = torch.zeros(extended.shape[0], 1, extended.shape[-1], device=extended.device, dtype=extended.dtype)
        if existing_mask_mode != "overwrite":
            new_mask[:, :, left_frames:left_frames + old_mask.shape[-1]] = old_mask
        if left_frames:
            new_mask[:, :, :left_frames] = 1.0
        if right_frames:
            new_mask[:, :, left_frames + samples.shape[-1]:] = 1.0

        out = _copy_latent(latent, extended)
        out["noise_mask"] = new_mask
        seconds = _frames_to_seconds(extended.shape[-1])
        summary = (
            f"source={_frames_to_seconds(samples.shape[-1]):.2f}s, "
            f"left={_frames_to_seconds(left_frames):.2f}s, "
            f"right={_frames_to_seconds(right_frames):.2f}s, output={seconds:.2f}s"
        )
        return (out, seconds, summary)


class ACEStep15XLExtendAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "vae": ("VAE",),
                "left_extend_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "right_extend_seconds": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "FLOAT", "STRING")
    RETURN_NAMES = ("latent", "source_latent", "seconds", "summary")
    FUNCTION = "extend"
    CATEGORY = CATEGORY

    def extend(self, audio, vae, left_extend_seconds: float, right_extend_seconds: float):
        source_latent = _encode_audio_with_vae(audio, vae)
        latent, seconds, summary = ACEStep15XLExtendLatent().extend(
            source_latent,
            left_extend_seconds,
            right_extend_seconds,
            "overwrite",
        )
        return (latent, source_latent, seconds, summary)


class ACEStep15XLRepaintLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "mask_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "existing_mask_mode": (["overwrite", "add", "subtract"], {"default": "overwrite"}),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "STRING")
    RETURN_NAMES = ("latent", "seconds", "summary")
    FUNCTION = "repaint"
    CATEGORY = CATEGORY

    def repaint(
        self,
        latent,
        start_seconds: float,
        end_seconds: float,
        mask_value: float,
        existing_mask_mode: str,
    ):
        samples = _latent_samples(latent)
        total_frames = samples.shape[-1]
        start_frame = max(0, min(total_frames, int(round(start_seconds * ACE15_LATENTS_PER_SECOND))))
        end_frame = total_frames if end_seconds <= 0 else int(round(end_seconds * ACE15_LATENTS_PER_SECOND))
        end_frame = max(0, min(total_frames, end_frame))
        if end_frame <= start_frame:
            raise ValueError("end_seconds must be greater than start_seconds, or set end_seconds to 0 to use the end.")

        mask = _mask_like(samples, latent, existing_mask_mode)
        mask = _apply_mask_mode(mask, start_frame, end_frame, mask_value, existing_mask_mode)

        out = _copy_latent(latent, samples)
        out["noise_mask"] = mask
        seconds = _frames_to_seconds(total_frames)
        summary = (
            f"masked={_frames_to_seconds(start_frame):.2f}s-"
            f"{_frames_to_seconds(end_frame):.2f}s, output={seconds:.2f}s, mask_value={mask_value:.2f}"
        )
        return (out, seconds, summary)


class ACEStep15XLRepaintAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "vae": ("VAE",),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "mask_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "FLOAT", "STRING")
    RETURN_NAMES = ("latent", "source_latent", "seconds", "summary")
    FUNCTION = "repaint"
    CATEGORY = CATEGORY

    def repaint(self, audio, vae, start_seconds: float, end_seconds: float, mask_value: float):
        source_latent = _encode_audio_with_vae(audio, vae)
        latent, seconds, summary = ACEStep15XLRepaintLatent().repaint(
            source_latent,
            start_seconds,
            end_seconds,
            mask_value,
            "overwrite",
        )
        return (latent, source_latent, seconds, summary)


class ACEStep15XLEditLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "edit_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "edit_end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "edit_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "existing_mask_mode": (["overwrite", "add", "subtract"], {"default": "overwrite"}),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "STRING")
    RETURN_NAMES = ("latent", "seconds", "summary")
    FUNCTION = "edit"
    CATEGORY = CATEGORY

    def edit(
        self,
        latent,
        edit_start_seconds: float,
        edit_end_seconds: float,
        edit_strength: float,
        existing_mask_mode: str,
    ):
        return ACEStep15XLRepaintLatent().repaint(
            latent,
            edit_start_seconds,
            edit_end_seconds,
            edit_strength,
            existing_mask_mode,
        )


class ACEStep15XLEditAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "vae": ("VAE",),
                "edit_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "edit_end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "edit_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "FLOAT", "STRING")
    RETURN_NAMES = ("latent", "source_latent", "seconds", "summary")
    FUNCTION = "edit"
    CATEGORY = CATEGORY

    def edit(self, audio, vae, edit_start_seconds: float, edit_end_seconds: float, edit_strength: float):
        source_latent = _encode_audio_with_vae(audio, vae)
        latent, seconds, summary = ACEStep15XLEditLatent().edit(
            source_latent,
            edit_start_seconds,
            edit_end_seconds,
            edit_strength,
            "overwrite",
        )
        return (latent, source_latent, seconds, summary)


NODE_CLASS_MAPPINGS = {
    "ACEStep15XLPromptLyrics": ACEStep15XLPromptLyrics,
    "ACEStep15XLTTSLikePrompt": ACEStep15XLTTSLikePrompt,
    "ACEStep15XLTextEncode": ACEStep15XLTextEncode,
    "ACEStep15XLEmptyLatentAudio": ACEStep15XLEmptyLatentAudio,
    "ACEStep15XLAudioToLatent": ACEStep15XLAudioToLatent,
    "ACEStep15XLReferenceAudio": ACEStep15XLReferenceAudio,
    "ACEStep15XLReferenceLatent": ACEStep15XLReferenceLatent,
    "ACEStep15XLExtendAudio": ACEStep15XLExtendAudio,
    "ACEStep15XLExtendLatent": ACEStep15XLExtendLatent,
    "ACEStep15XLRepaintAudio": ACEStep15XLRepaintAudio,
    "ACEStep15XLRepaintLatent": ACEStep15XLRepaintLatent,
    "ACEStep15XLEditAudio": ACEStep15XLEditAudio,
    "ACEStep15XLEditLatent": ACEStep15XLEditLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACEStep15XLPromptLyrics": "🎵 ACE-Step 1.5XL Prompt + Lyrics",
    "ACEStep15XLTTSLikePrompt": "🎵 ACE-Step 1.5XL TTS-like Voice Prompt",
    "ACEStep15XLTextEncode": "🎵 ACE-Step 1.5XL Text Encode",
    "ACEStep15XLEmptyLatentAudio": "🎵 ACE-Step 1.5XL Empty Latent Audio",
    "ACEStep15XLAudioToLatent": "🎵 ACE-Step 1.5XL Audio to Latent",
    "ACEStep15XLReferenceAudio": "🎵 ACE-Step 1.5XL Reference Audio",
    "ACEStep15XLReferenceLatent": "🎵 ACE-Step 1.5XL Reference Latent",
    "ACEStep15XLExtendAudio": "🎵 ACE-Step 1.5XL Extend Audio",
    "ACEStep15XLExtendLatent": "🎵 ACE-Step 1.5XL Extend Latent",
    "ACEStep15XLRepaintAudio": "🎵 ACE-Step 1.5XL Repaint Audio",
    "ACEStep15XLRepaintLatent": "🎵 ACE-Step 1.5XL Repaint Latent",
    "ACEStep15XLEditAudio": "🎵 ACE-Step 1.5XL Edit Audio",
    "ACEStep15XLEditLatent": "🎵 ACE-Step 1.5XL Edit Latent",
}
