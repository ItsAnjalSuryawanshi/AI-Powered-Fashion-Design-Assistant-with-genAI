import os
import torch
import logging
import warnings
from PIL import Image
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from transformers import CLIPProcessor, CLIPModel

# Silence deprecation warnings and logs for a clean console output
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("AarivyaLabsEngine")

# Disable extra verbose logging from huggingface models
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

class FashionGenAIPipeline:
    def __init__(self):
        """
        Initializes the dynamic generation and evaluation engine.
        Automatically detects and optimizes code for GPU (CUDA) or CPU execution.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use float16 precision on GPU for speed; float32 on CPU for stability
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        logger.info(f"System active. Target Hardware Device: {self.device.upper()}")
        
        self.diffusion_model_id = "runwayml/stable-diffusion-v1-5"
        self.clip_model_id = "openai/clip-vit-base-patch32"
        
        self.pipe = None
        self.clip_model = None
        self.clip_processor = None

    def load_models(self):
        """Loads and applies memory-saving optimizations to the neural networks."""
        if not self.pipe:
            logger.info("Loading Stable Diffusion Core Engine (this may take a few minutes on first run)...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.diffusion_model_id, 
                torch_dtype=self.dtype
            ).to(self.device)
            
            # Use UniPCMultistepScheduler for clean, crisp images in fewer inference steps
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            # Optimization Matrix: Prevents systems from running out of VRAM/RAM
            if self.device == "cpu":
                logger.warning("Running on CPU. Activating memory slice optimizations...")
                self.pipe.enable_attention_slicing()
            else:
                # Safe xformers fallback block to handle environments without the library installed
                try:
                    if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                        self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    logger.warning("xformers package not detected. Falling back to native PyTorch optimization heads.")
                    pass
                
        if not self.clip_model:
            logger.info("Loading CLIP Metric Evaluation Suite...")
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_id).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
            
        logger.info("All neural networks loaded successfully.")

    def generate_brand_style(self, base_prompt: str, brand_token: str) -> Image.Image:
        """
        Executes token-based style switching. Modifies the prompt matrix 
        to steer the generative model toward specific brand identities.
        """
        self.load_models()
        
        # High-fashion prompt framing
        full_prompt = f"high fashion runway photography, {base_prompt}, styled in {brand_token} aesthetic, lookbook style, photorealistic, 8k resolution"
        negative_prompt = "deformed anatomy, blurry, low quality, sketch, cartoon, text, watermark, bad proportions, bad hands"
        
        logger.info(f"Dispatching Inference Pipeline for Token [{brand_token}]...")
        
        # Run inference using inference_mode to stop tracking gradients and save huge memory
        with torch.inference_mode():
            image = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=20, # Efficiently balanced generation steps
                guidance_scale=7.5
            ).images[0]
            
        return image

    def evaluate_clip_score(self, image: Image.Image, evaluation_concept: str) -> float:
        """
        Extracts cross-modal alignment scores using CLIP's native logit scale.
        Fixed to prevent pooling wrapper attribute errors across different library versions.
        """
        self.load_models()
        
        inputs = self.clip_processor(
            text=[evaluation_concept], 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            # Use the official forward pass to pull cross-modal alignment logits directly
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            
            # CLIP internally scales scores by 100, divide to return standard cosine similarity
            cosine_similarity = logits_per_image.item() / 100.0
            
        return round(cosine_similarity, 4)


# Main pipeline execution environment
if __name__ == "__main__":
    print("="*70)
   
    print("="*70)
    
    # Initialize pipeline object
    engine = FashionGenAIPipeline()
    
    # -----------------------------------------------------------------
    # RUNTIME TEST 1: Cyberpunk Techwear Identity Matrix
    # -----------------------------------------------------------------
    prompt_1 = "a modular utility cargo jacket with industrial buckles and straps"
    token_1 = "cybpng style, dark futuristic techwear clothing"
    
    img_1 = engine.generate_brand_style(prompt_1, token_1)
    img_1.save("style_output_cyberpunk.png")
    
    score_1 = engine.evaluate_clip_score(img_1, "futuristic cyberpunk tactical techwear garment design")
    logger.info(f"Execution [1/2] Complete. Cyberpunk CLIP Similarity Score: {score_1}")
    
    # -----------------------------------------------------------------
    # RUNTIME TEST 2: Minimalist Haute Couture Identity Matrix
    # -----------------------------------------------------------------
    prompt_2 = "an asymmetrical flowing luxury evening dress drape"
    token_2 = "mnmlst style, premium clean monochromatic fluid silk fashion"
    
    img_2 = engine.generate_brand_style(prompt_2, token_2)
    img_2.save("style_output_minimalist.png")
    
    score_2 = engine.evaluate_clip_score(img_2, "clean minimalist elegant high fashion couture dress")
    logger.info(f"Execution [2/2] Complete. Minimalist CLIP Similarity Score: {score_2}")
    
    print("\n" + "="*70)
    print(" SUCCESS")
    print("="*70)