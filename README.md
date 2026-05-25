# ComfyUI-ACEStep

Custom ACE-Step 1.5XL helper nodes for native ComfyUI audio workflows.

The nodes are prefixed with `ACEStep15XL` so they can coexist with other ACE-Step custom nodes.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory, then restart ComfyUI.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/starsFriday/ComfyUI-ACEStep.git
```

## Model Files

The example workflows expect the standard ACE-Step 1.5XL model files to be available in ComfyUI:

- `diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors`
- `clip/qwen_0.6b_ace15.safetensors`
- `clip/qwen_4b_ace15.safetensors`
- `vae/ace_1.5_vae.safetensors`

Use the filenames and folders configured by your ComfyUI installation if they differ.

## Nodes

- `ACE-Step 1.5XL Prompt + Lyrics`: simple tags and lyrics passthrough node.
- `ACE-Step 1.5XL TTS-like Voice Prompt`: formats a script into ACE lyrics and outputs tags, bpm, duration, language, time signature, and keyscale for direct connection to the text encoder.
- `ACE-Step 1.5XL Text Encode`: text encoder wrapper matching ComfyUI's native `TextEncodeAceStepAudio1.5` behavior.
- `ACE-Step 1.5XL Empty Latent Audio`: creates an empty ACE 1.5 audio latent at 25 latent frames per second.
- `ACE-Step 1.5XL Audio to Latent`: encodes `AUDIO` with the ACE 1.5 VAE.
- `ACE-Step 1.5XL Reference Audio`: attaches reference audio timbre latents to positive conditioning and outputs reference-free negative conditioning.
- `ACE-Step 1.5XL Reference Latent`: same as reference audio, but accepts a pre-encoded latent.
- `ACE-Step 1.5XL Extend Audio/Latent`: pads audio latents with ACE silence and masks the new region for generation.
- `ACE-Step 1.5XL Repaint Audio/Latent`: masks a time range for regeneration.
- `ACE-Step 1.5XL Edit Audio/Latent`: masks a time range for regeneration with changed tags or lyrics.

## Example Workflows

Example workflows are stored in `workflows/`:

- `ace_step_15xl_text_generation.json`
- `ace_step_15xl_reference_audio.json`
- `ace_step_15xl_extend_audio.json`
- `ace_step_15xl_edit_audio.json`
- `ace_step_15xl_repaint_audio.json`

The audio workflows use placeholder filenames such as `source_audio.wav` and `reference.wav`. After importing a workflow, select your own audio file in each `Load Audio` node.

## Reference Audio

1. Build positive conditioning with `ACE-Step 1.5XL Text Encode`.
2. Send it through `ACE-Step 1.5XL Reference Audio`.
3. Connect `conditioning` to KSampler positive.
4. Connect `negative_conditioning` to KSampler negative.
5. Keep `generate_audio_codes` disabled when the reference voice should dominate.

Do not run `ConditioningZeroOut` after the reference node for negative conditioning because it can keep the reference payload and weaken the voice reference effect.

## TTS-like Reference Voice

The reference workflow can be used for TTS-like voice-guided singing or narration:

1. Open `ace_step_15xl_reference_audio.json`.
2. Put the target text into `ACE-Step 1.5XL TTS-like Voice Prompt`.
3. Choose a clean vocal sample in `Load Audio`; 10-30 seconds usually works better than mixed music.
4. Keep `generate_audio_codes` disabled so the reference voice can steer the vocal timbre.

This is not strict TTS. ACE-Step can still reinterpret timing, melody, pronunciation, and phrasing.

## Extend, Edit, And Repaint

For extension, load or generate source audio, use `ACE-Step 1.5XL Extend Audio`, connect the returned latent to KSampler, and set text encode duration from the returned `seconds` value.

For edit and repaint, use `ACE-Step 1.5XL Edit Audio` or `ACE-Step 1.5XL Repaint Audio`, connect the returned latent to KSampler, and adjust KSampler `denoise` together with the node mask settings.

## Appearance

`web/ace_appearance.js` sets the default color for ACE-Step nodes created from search or menu. Saved workflows keep their serialized `color` and `bgcolor` values.
