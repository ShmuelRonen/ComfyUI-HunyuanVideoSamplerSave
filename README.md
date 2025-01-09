# ComfyUI-HunyuanVideoSamplerSave

A ComfyUI custom node implementation for optimized video generation and motion effects, designed to work with Hunyuan text-to-video models.

Image to Video:
![image](https://github.com/user-attachments/assets/d1acf721-339a-41e4-b757-c680e758939d)


## Features

### HunyuanVideoSamplerSave
An optimized video sampler that extends ComfyUI's KSampler capabilities:
- Memory-efficient batch processing for video frames
- Progress tracking for long video generation tasks
- Optimized VRAM usage through sequential frame processing
- Interrupt-safe with proper memory management
- Compatible with all standard ComfyUI samplers and schedulers

### ImageMotionInfluence
A powerful tool for creating motion sequences from static images:
- Horizontal panning effects with adjustable range
- Progressive zoom capabilities
- Seamless loop generation through mirror techniques
- Configurable frame count and motion parameters

### ResizeImageForHunyuan
A specialized resizing tool optimized for Hunyuan video generation:
- Predefined aspect ratios optimized for home GPUs
- Multiple size options for each aspect ratio
- All dimensions properly aligned to 16x16 grid
- Multiple upscaling methods
- Crop control options

### EmptyVideoLatentForHunyuan
A latent initialization tool specifically designed for Hunyuan video generation:
- Supports multiple optimized resolutions for home GPUs
- Common aspect ratios (16:9, 4:3, 3:2, 9:16, 3:4, 2:3)
- Memory-efficient latent generation
- Configurable video length and batch size
- All dimensions automatically aligned to model requirements

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/ShmuelRonen/ComfyUI-HunyuanVideoSamplerSave.git
```

2. Restart ComfyUI to load the new nodes.

## Usage

### Video Generation Workflow

1. **Image Motion Setup**
   - Input: Any source image
   - Configure motion parameters:
     - move_range_x: Controls horizontal movement (-150 to 150)
     - frame_num: Number of frames to generate (2 to 500)
     - zoom: Progressive zoom effect (0.0 to 0.5)
   - Output: Sequence of motion-affected images

2. **Image Resizing**
   - Use ResizeImageForHunyuan to ensure proper dimensions
   - Select from optimized presets for your GPU
   - Choose appropriate upscaling method

3. **Latent Setup**
   - Use EmptyVideoLatentForHunyuan to initialize latent space
   - Select resolution from optimized presets
   - Configure video length and batch size

4. **Video Generation**
   - Use HunyuanVideoSamplerSave with your text prompts
   - The motion-influenced latents guide the video generation
   - Adjustable parameters:
     - Steps: Generation steps per frame
     - CFG: Prompt influence strength
     - Sampler and Scheduler selection
     - Denoising strength

## Parameters

### HunyuanVideoSamplerSave
- **model**: Loaded Hunyuan model
- **positive/negative**: Conditioning from text prompts
- **video_latents**: Input latent sequence
- **seed**: Generation seed for reproducibility
- **steps**: Number of sampling steps
- **cfg**: Conditioning strength
- **sampler_name**: Choice of sampling algorithm
- **scheduler**: Noise scheduler selection
- **denoise**: Denoising strength

### ImageMotionInfluence
- **image**: Input source image
- **move_range_x**: Horizontal motion range
- **frame_num**: Number of frames to generate
- **zoom**: Zoom effect intensity

### ResizeImageForHunyuan
- **image**: Input image to resize
- **size_preset**: Selection of predefined sizes (e.g., "384x216 (16:9)", "768x432 (16:9)")
- **upscale_method**: Choice of upscaling algorithm (nearest-exact, bilinear, area, bicubic)
- **crop**: Crop method selection (disabled, center)

### EmptyVideoLatentForHunyuan
- **resolution**: Selection of optimized video resolutions
- **length**: Number of frames to generate
- **batch_size**: Number of videos to generate in parallel

## Memory Optimization

The nodes implement several memory optimization strategies:
- Sequential frame processing
- Active memory management
- Intermediate result storage
- Garbage collection during processing
- Optimized resolution presets for home GPUs
- Proper dimension alignment for efficient processing

This allows for processing of longer sequences and higher resolution outputs compared to standard sampling approaches.

## Integration

This custom node is designed to work seamlessly with:
- ComfyUI's core components
- Hunyuan text-to-video models
- Standard VAE encoders
- Various sampling and scheduling methods

## Requirements

- ComfyUI installation
- Compatible Hunyuan model
- Sufficient VRAM for video processing
- Python 3.x
- PyTorch

## License

MIT
