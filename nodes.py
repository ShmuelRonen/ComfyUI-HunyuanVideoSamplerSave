import torch
from torch import nn
import comfy.sample
import comfy.model_management
import comfy.utils
import gc
import logging
import nodes

class ImageMotionInfluance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "image": ("IMAGE",),
            "move_range_x": ("INT", {"default": 0, "min": -150, "max": 150}),
            "frame_num": ("INT", {"default": 10, "min": 2, "max": 500}),
            "zoom": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.05}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "guide_motion"
    CATEGORY = "image/animation"

    def get_size(self, image):
        image_size = image.size()
        return int(image_size[2]), int(image_size[1])

    def guide_motion(self, image, move_range_x, frame_num, zoom):
        img_width, img_height = self.get_size(image)
        
        step_size = abs(move_range_x) / (frame_num - 1) if move_range_x != 0 else 0
        start_x = 0 if move_range_x > 0 else abs(move_range_x)
        
        if move_range_x < 0:
            start_x -= img_width
        
        batch = []
        mirrored = torch.flip(image, [2])
        
        for i in range(frame_num):
            x_pos = start_x + (step_size * i * (-1 if move_range_x < 0 else 1))
            x_pos = int(x_pos)
            x_pos = x_pos % img_width if move_range_x != 0 else 0
            
            current_zoom = (i / (frame_num - 1)) * zoom if zoom > 0 else 0
            if current_zoom > 0:
                crop_width = int(img_width * (1 - current_zoom))
                crop_height = int(img_height * (1 - current_zoom))
                x_start = (img_width - crop_width) // 2
                y_start = (img_height - crop_height) // 2
                
                zoomed_original = torch.nn.functional.interpolate(
                    image[:, y_start:y_start + crop_height, x_start:x_start + crop_width, :].permute(0, 3, 1, 2),
                    size=(img_height, img_width),
                    mode='bilinear'
                ).permute(0, 2, 3, 1)
                
                zoomed_mirror = torch.nn.functional.interpolate(
                    mirrored[:, y_start:y_start + crop_height, x_start:x_start + crop_width, :].permute(0, 3, 1, 2),
                    size=(img_height, img_width),
                    mode='bilinear'
                ).permute(0, 2, 3, 1)
            else:
                zoomed_original = image
                zoomed_mirror = mirrored
            
            canvas = torch.zeros((1, img_height, img_width, image.shape[3]))
            
            remaining_width = img_width
            current_x = x_pos
            use_flipped = False
            
            while remaining_width > 0:
                width = min(img_width - current_x, remaining_width)
                current_image = zoomed_mirror if use_flipped else zoomed_original
                
                canvas[0, :, img_width - remaining_width:img_width - remaining_width + width, :] = \
                    current_image[0, :, current_x:current_x + width, :]
                
                remaining_width -= width
                current_x = 0
                use_flipped = not use_flipped
            
            batch.append(canvas)
            
        return (torch.cat(batch, dim=0),)

class MotionGuidedSampler(nn.Module):
    def __init__(
        self,
        motion_strength: float = 0.5,
        consistency_strength: float = 0.9,
        denoise_strength: float = 0.8
    ):
        super().__init__()
        self.motion_strength = motion_strength
        self.consistency_strength = consistency_strength
        self.denoise_strength = denoise_strength
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_motion_vector(self, current_latent: torch.Tensor, prev_latent: torch.Tensor) -> torch.Tensor:
        try:
            return current_latent - prev_latent
        except RuntimeError as e:
            self.logger.error(f"Motion extraction error: {e}")
            return torch.zeros_like(current_latent)

    def apply_motion_vector(self, content: torch.Tensor, motion_vector: torch.Tensor) -> torch.Tensor:
        try:
            motion_applied = content + (motion_vector * self.motion_strength)
            if torch.isnan(motion_applied).any():
                self.logger.warning("NaN detected in motion application")
                return content
            return motion_applied
        except RuntimeError as e:
            self.logger.error(f"Motion application error: {e}")
            return content

    def process_frames(self, latent_frames, model, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise):
        batch_size = latent_frames.shape[0]
        processed_frames = []
        device = comfy.model_management.get_torch_device()
        
        latent_frames = latent_frames.to(device)
        pbar = comfy.utils.ProgressBar(batch_size)
        
        # Initialize sampler
        sampler = comfy.samplers.KSampler(
            model, 
            steps=steps,
            device=device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=denoise
        )

        # Process first frame
        noise = comfy.sample.prepare_noise(latent_frames[0:1], seed, None).to(device)
        first_frame = sampler.sample(
            noise,
            positive,
            negative,
            cfg=cfg,
            latent_image=latent_frames[0:1],
            force_full_denoise=True
        )
        
        processed_frames.append(first_frame)
        prev_orig = latent_frames[0:1].to(device)
        prev_styled = first_frame
        
        pbar.update(1)

        # Process remaining frames
        for i in range(1, batch_size):
            current_orig = latent_frames[i:i+1].to(device)
            
            motion = self.extract_motion_vector(current_orig, prev_orig)
            motion_guided = self.apply_motion_vector(prev_styled, motion)
            
            current_noise = comfy.sample.prepare_noise(motion_guided, seed + i, None).to(device)
            current_frame = sampler.sample(
                current_noise,
                positive,
                negative,
                cfg=cfg,
                latent_image=motion_guided,
                force_full_denoise=True
            )
            
            processed_frames.append(current_frame)
            prev_orig = current_orig
            prev_styled = current_frame
            
            pbar.update(1)
            
            if i % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                self.logger.info(f"Processed {i}/{batch_size} frames")
        
        result = torch.cat(processed_frames, dim=0)
        return result.to(device)

    def forward(self, latent_frames, model, positive, negative, noise_seed, steps, cfg, sampler_name, scheduler, denoise):
        try:
            return self.process_frames(
                latent_frames=latent_frames,
                model=model,
                positive=positive,
                negative=negative,
                seed=noise_seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise
            )
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            raise e

class HunyuanVideoSamplerSave:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "video_latents": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, video_latents, positive, negative, seed, steps,
               cfg, sampler_name, scheduler, denoise):
        device = comfy.model_management.get_torch_device()
        
        # Print latent size for debugging
        print(f"Original latent size: {video_latents['samples'].size()}")
        
        # Use nodes.common_ksampler which works with hunyuan
        return nodes.common_ksampler(
            model, 
            seed, 
            steps, 
            cfg, 
            sampler_name, 
            scheduler, 
            positive, 
            negative, 
            video_latents,  # Pass the entire latent dict
            denoise=denoise
        )

class ResizeImageForHunyuan:
    crop_methods = ["disabled", "center"]
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]
    
    # Define all supported aspect ratios and their sizes
    size_presets = {
        "16:9": [(384,216), (512,288), (640,360), (768,432)],
        "4:3": [(384,288), (512,384), (640,480), (768,576)],
        "3:2": [(384,256), (512,336), (640,432), (768,512)],
        "9:16": [(216,384), (288,512), (360,640), (432,768)],
        "3:4": [(288,384), (384,512), (480,640), (576,768)],
        "2:3": [(256,384), (336,512), (432,640), (512,768)]
    }

    @classmethod
    def INPUT_TYPES(s):
        # Create size options list
        size_options = []
        for ratio, sizes in s.size_presets.items():
            for w, h in sizes:
                size_options.append(f"{ratio} ({w}x{h})")

        return {"required": {
            "image": ("IMAGE",),
            "size_preset": (size_options,),
            "upscale_method": (s.upscale_methods,),
            "crop": (s.crop_methods,)
        }}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'resize'
    CATEGORY = 'image/hunyuan'

    def parse_size_preset(self, preset):
        # Extract width and height from preset string like "16:9 (768x432)"
        size_part = preset.split(" ")[1].strip("()")
        width, height = map(int, size_part.split("x"))
        return width, height

    def upscale(self, image, upscale_method, width, height, crop):
        samples = image.movedim(-1,1)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        s = s.movedim(1,-1)
        return (s,)

    def resize(self, image, size_preset, upscale_method, crop):
        # Get target width and height from preset
        w, h = self.parse_size_preset(size_preset)
        
        # Print info about the resize operation
        print(f'Resizing image from {image.shape[2]}x{image.shape[1]} to {w}x{h} ({size_preset.split()[0]} aspect ratio)')
        
        img = self.upscale(image, upscale_method, w, h, crop)[0]
        return (img, )

class EmptyVideoLatentForHunyuan:
    @classmethod
    def INPUT_TYPES(s):
        # Define aspect ratios for Hunyuan video that can work with home GPUs (max width 768)
        aspect_ratios = [
            "384x216 (16:9)", 
            "512x288 (16:9)",
            "640x360 (16:9)",
            "768x432 (16:9)",
            "384x288 (4:3)",
            "512x384 (4:3)",
            "640x480 (4:3)",
            "768x576 (4:3)",
            "384x256 (3:2)",
            "512x336 (3:2)",
            "640x432 (3:2)",
            "768x512 (3:2)",
            "216x384 (9:16)",
            "288x512 (9:16)",
            "360x640 (9:16)",
            "432x768 (9:16)",
            "288x384 (3:4)",
            "384x512 (3:4)",
            "480x640 (3:4)",
            "576x768 (3:4)",
            "256x384 (2:3)",
            "336x512 (2:3)",
            "432x640 (2:3)",
            "512x768 (2:3)"
        ]
        
        return {"required": { 
            "resolution": (aspect_ratios, ),
            "length": ("INT", {"default": 25, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
        }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent/video"

    def generate(self, resolution, length, batch_size=1):
        # Parse resolution string to get dimensions
        dimensions = resolution.split(' ')[0]
        width, height = map(int, dimensions.split('x'))
        
        # Ensure dimensions are multiples of 16
        width = (width // 16) * 16
        height = (height // 16) * 16
        
        # Create latent with original format
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                           device=comfy.model_management.intermediate_device())
        return ({"samples": latent}, )

# Node registration
NODE_CLASS_MAPPINGS = {
    "HunyuanVideoSamplerSave": HunyuanVideoSamplerSave,
    "ResizeImageForHunyuan": ResizeImageForHunyuan,
    "EmptyVideoLatentForHunyuan": EmptyVideoLatentForHunyuan
    "ImageMotionInfluance": ImageMotionInfluance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanVideoSamplerSave": "Hunyuan Video Sampler Save",
    "ResizeImageForHunyuan": "Resize Image For Hunyuan",
    "EmptyVideoLatentForHunyuan": "Empty Video Latent For Hunyuan",
    "ImageMotionInfluance": "Image Motion Influence",
}

