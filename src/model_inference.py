"""Self-contained model inference module for T2F demo."""

import sys
import importlib
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Optional
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np


# ============================================================================
# TEXT ENCODER
# ============================================================================

class TextEncoder(nn.Module):
    """BERT-based text encoder."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        max_length: int = 128,
        freeze: bool = True,
        pooling: str = "cls"
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.freeze = freeze
        self.pooling = pooling.lower()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            self.bert_model.eval()
        
        hidden_size = self.bert_model.config.hidden_size
        
        if self.pooling == "concat":
            pooled_dim = hidden_size * 3
        else:
            pooled_dim = hidden_size
        
        self.projection = None if embedding_dim == pooled_dim else nn.Linear(pooled_dim, embedding_dim)
    
    def _pool_tokens(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        """Pool token embeddings."""
        mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
        masked = token_embeddings * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)
        mean_embedding = masked.sum(dim=1) / denom
        masked_for_max = token_embeddings.masked_fill(mask == 0, float("-inf"))
        max_embedding = masked_for_max.max(dim=1).values
        return mean_embedding, max_embedding
    
    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text(s) to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        if self.freeze:
            with torch.enable_grad():
                outputs = self.bert_model(**inputs)
                
                if self.pooling == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooling == "mean":
                    mean_embedding, _ = self._pool_tokens(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
                    embeddings = mean_embedding
                elif self.pooling == "max":
                    _, max_embedding = self._pool_tokens(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
                    embeddings = max_embedding
                else:  # concat
                    cls_embedding = outputs.pooler_output
                    mean_embedding, max_embedding = self._pool_tokens(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
                    embeddings = torch.cat([cls_embedding, mean_embedding, max_embedding], dim=1)
                
                if self.projection:
                    embeddings = self.projection(embeddings)
            
            return embeddings
        else:
            outputs = self.bert_model(**inputs)
            
            if self.pooling == "cls":
                embeddings = outputs.pooler_output
            elif self.pooling == "mean":
                mean_embedding, _ = self._pool_tokens(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings = mean_embedding
            elif self.pooling == "max":
                _, max_embedding = self._pool_tokens(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings = max_embedding
            else:  # concat
                cls_embedding = outputs.pooler_output
                mean_embedding, max_embedding = self._pool_tokens(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings = torch.cat([cls_embedding, mean_embedding, max_embedding], dim=1)
            
            if self.projection:
                embeddings = self.projection(embeddings)
            
            return embeddings


# ============================================================================
# MAPPER NETWORK
# ============================================================================

def _activation(name: str):
    """Get activation function by name."""
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    if name.lower() == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class TextToLatentMapper(nn.Module):
    """MLP network that maps text embeddings to StyleGAN2 latent codes."""
    
    def __init__(
        self,
        input_dim: int = 768,
        intermediate_dims: List[int] = [1024, 1024],
        output_dim: int = 512,
        w_plus_layers: int = 18,
        latent_space: str = "wplus",
        activation: str = "relu",
        dropout: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.w_plus_layers = w_plus_layers
        self.latent_space = latent_space.lower()
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        
        layers = []
        
        def add_block(in_dim: int, out_dim: int):
            layers.append(nn.Linear(in_dim, out_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            elif use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        prev_dim = input_dim
        for dim in intermediate_dims:
            add_block(prev_dim, dim)
            prev_dim = dim
        
        if self.latent_space == "z":
            final_output_dim = 512
        elif self.latent_space == "wplus":
            final_output_dim = self.w_plus_layers * self.output_dim
        else:  # "w"
            final_output_dim = self.output_dim
        
        layers.append(nn.Linear(prev_dim, final_output_dim))
        self.mapping_network = nn.Sequential(*layers)
    
    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Map text embeddings to latent codes."""
        original_batch_size = text_embeddings.size(0)
        
        if self.use_batch_norm and not self.use_layer_norm and original_batch_size == 1:
            text_embeddings = text_embeddings.repeat(2, 1)
            w_flat = self.mapping_network(text_embeddings)
            w_flat = w_flat[:1]
        else:
            w_flat = self.mapping_network(text_embeddings)
        
        batch_size = original_batch_size
        
        if self.latent_space == "z":
            return w_flat.view(batch_size, -1)
        elif self.latent_space == "wplus":
            return w_flat.view(batch_size, self.w_plus_layers, self.output_dim)
        else:  # "w"
            return w_flat.view(batch_size, self.output_dim)


# ============================================================================
# STYLEGAN2 WRAPPER
# ============================================================================

class StyleGAN2Wrapper(nn.Module):
    """Wrapper for pretrained StyleGAN2 generator."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        truncation_psi: float = 0.7,
        noise_mode: str = "const",
        stylegan_root: Optional[str] = None,
        latent_space: str = "wplus",
    ):
        super().__init__()
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = torch.device(device)
        self.truncation_psi = truncation_psi
        self.noise_mode = noise_mode
        self.w_center = None
        self.latent_space = latent_space.lower()
        
        self._prepare_stylegan_modules(stylegan_root)
        self._load_model()
        
        if self.generator.w_dim != 512:
            self.w_projection = nn.Linear(512, self.generator.w_dim).to(self.device)
        else:
            self.w_projection = None
        
        self.w_center = torch.zeros(1, 512, device=self.device)
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def _prepare_stylegan_modules(self, stylegan_root: Optional[str]):
        """Add StyleGAN2 directory to Python path."""
        if stylegan_root is None:
            # Try to find stylegan2-ada-pytorch in current directory (demo folder)
            current_dir = Path(__file__).parent.resolve()
            stylegan_root = current_dir / "stylegan2-ada-pytorch"
            
            # If not found, try parent directory
            if not stylegan_root.exists():
                stylegan_root = current_dir.parent / "stylegan2-ada-pytorch"
        
        stylegan_root = Path(stylegan_root).expanduser().resolve()
        
        if not stylegan_root.exists():
            raise ImportError(
                f"Cannot find stylegan2-ada-pytorch directory. "
                f"Expected at: {stylegan_root}\n"
                f"Please ensure stylegan2-ada-pytorch is cloned in the project directory."
            )
        
        if str(stylegan_root) not in sys.path:
            sys.path.insert(0, str(stylegan_root))
        
        try:
            importlib.import_module("dnnlib")
            importlib.import_module("legacy")
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"Cannot find stylegan2-ada-pytorch modules. "
                f"Path exists: {stylegan_root}\n"
                f"But failed to import dnnlib/legacy modules.\n"
                f"Error: {str(exc)}"
            ) from exc
    
    def _load_model(self):
        """Load pretrained StyleGAN2 generator."""
        model_path = self.model_path
        
        # Try multiple locations for ffhq.pkl
        if not model_path.exists():
            project_root = Path(__file__).parent.parent
            alt_paths = [
                project_root / "models" / "ffhq.pkl",
                project_root / "pretrained_model" / "ffhq.pkl",
                project_root.parent / "t2f_training" / "pretrained_model" / "ffhq.pkl",
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_path = alt_path
                    print(f"✓ Found StyleGAN2 at: {model_path}")
                    break
            else:
                raise FileNotFoundError(
                    f"StyleGAN2 model (ffhq.pkl) not found.\n\n"
                    f"Tried locations:\n"
                    f"  - {self.model_path}\n" +
                    "\n".join(f"  - {p}" for p in alt_paths) +
                    f"\n\nOptions:\n"
                    f"  1. Copy from training project:\n"
                    f"     cp ../t2f_training/pretrained_model/ffhq.pkl models/\n"
                    f"  2. Download (350 MB):\n"
                    f"     wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -P models/"
                )
        
        import legacy
        
        with open(model_path, "rb") as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(self.device).eval()
        
        self.generator = G
        self.latent_dim = G.z_dim
        self.resolution = G.img_resolution
    
    def generate_from_latent(self, latent_codes: torch.Tensor, noise_mode: Optional[str] = None) -> torch.Tensor:
        """Generate images from latent codes."""
        noise_mode = self.noise_mode if noise_mode is None else noise_mode
        
        latent_codes = latent_codes.to(self.device, dtype=torch.float32)
        
        if self.latent_space == "z":
            w = self.generator.mapping(latent_codes, None)
            images = self.generator.synthesis(w, noise_mode=noise_mode)
        else:  # "w" or "wplus"
            if self.w_projection is not None:
                original_shape = latent_codes.shape
                if latent_codes.dim() == 3:
                    latent_codes = latent_codes.view(-1, latent_codes.shape[-1])
                
                latent_codes = self.w_projection(latent_codes)
                
                if len(original_shape) == 3:
                    latent_codes = latent_codes.view(original_shape[0], original_shape[1], -1)
            
            if latent_codes.dim() == 2:
                latent_codes = latent_codes.unsqueeze(1).repeat(1, self.generator.num_ws, 1)
            elif latent_codes.dim() == 3:
                if latent_codes.shape[1] != self.generator.num_ws:
                    if latent_codes.shape[1] < self.generator.num_ws:
                        repeat_factor = self.generator.num_ws // latent_codes.shape[1]
                        remainder = self.generator.num_ws % latent_codes.shape[1]
                        latent_codes = latent_codes.repeat(1, repeat_factor, 1)
                        if remainder > 0:
                            latent_codes = torch.cat([
                                latent_codes,
                                latent_codes[:, -remainder:, :].repeat(1, 1, 1)
                            ], dim=1)
                    else:
                        latent_codes = latent_codes[:, :self.generator.num_ws, :]
            
            images = self.generator.synthesis(latent_codes, noise_mode=noise_mode)
        
        return images
    
    def truncate_w(self, w: torch.Tensor, truncation_psi: float) -> torch.Tensor:
        """Apply truncation trick to W latents."""
        if truncation_psi == 1.0 or self.w_center is None or self.latent_space == "z":
            return w
        
        if w.dim() == 2:
            return self.w_center + truncation_psi * (w - self.w_center)
        w_center_expanded = self.w_center.unsqueeze(1).expand_as(w)
        return w_center_expanded + truncation_psi * (w - w_center_expanded)


# ============================================================================
# T2F MODEL
# ============================================================================

class T2FModel(nn.Module):
    """Complete Text-to-Face generation model."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
        model_cfg = config.get("model", {})
        text_encoder_cfg = model_cfg.get("text_encoder", {})
        mapper_cfg = model_cfg.get("mapper", {})
        stylegan_cfg = model_cfg.get("stylegan2", {})
        
        self.latent_space = model_cfg.get("latent_space", "wplus").lower()
        
        # Text encoder (on CPU)
        encoder_device = torch.device(text_encoder_cfg.get("device", "cpu"))
        self.text_encoder = TextEncoder(
            model_name=text_encoder_cfg.get("model_name", "bert-base-uncased"),
            embedding_dim=text_encoder_cfg.get("embedding_dim", 768),
            max_length=text_encoder_cfg.get("max_length", 128),
            freeze=text_encoder_cfg.get("freeze", True),
            pooling=text_encoder_cfg.get("pooling", "cls"),
        ).to(encoder_device)
        self.text_encoder_device = encoder_device
        
        # Mapper (trainable)
        self.mapper = TextToLatentMapper(
            input_dim=mapper_cfg.get("input_dim", text_encoder_cfg.get("embedding_dim", 768)),
            intermediate_dims=mapper_cfg.get("intermediate_dims", [1024, 1024]),
            output_dim=mapper_cfg.get("output_dim", 512),
            w_plus_layers=stylegan_cfg.get("w_plus_layers", 18),
            latent_space=self.latent_space,
            activation=mapper_cfg.get("activation", "relu"),
            dropout=mapper_cfg.get("dropout", 0.1),
            use_batch_norm=mapper_cfg.get("use_batch_norm", False),
            use_layer_norm=mapper_cfg.get("use_layer_norm", True),
        ).to(self.device)
        
        # StyleGAN2 wrapper (frozen)
        self.generator = StyleGAN2Wrapper(
            model_path=stylegan_cfg["model_path"],
            device=str(self.device),
            truncation_psi=stylegan_cfg.get("truncation_psi", 0.7),
            noise_mode=stylegan_cfg.get("noise_mode", "const"),
            stylegan_root=stylegan_cfg.get("stylegan_root"),
            latent_space=self.latent_space,
        )
        self.generator_device = self.generator.device
        self.truncation_psi = stylegan_cfg.get("truncation_psi", 0.7)
    
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text to embeddings."""
        embeddings = self.text_encoder(texts)
        return embeddings.to(self.device)
    
    def map_to_latent(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Map text embeddings to latent codes."""
        return self.mapper(text_embeddings.to(self.device))
    
    def generate_images(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """Generate images from latent codes."""
        latent_codes = latent_codes.to(self.generator_device)
        return self.generator.generate_from_latent(latent_codes)
    
    def forward(
        self,
        texts: Union[str, List[str]],
        return_latents: bool = False,
    ):
        """Complete forward pass: text -> latent -> image."""
        text_embeddings = self.encode_text(texts)
        latent_codes = self.map_to_latent(text_embeddings)
        
        if latent_codes.dtype != torch.float32:
            latent_codes = latent_codes.float()
        
        if self.latent_space != "z" and self.truncation_psi < 1.0:
            latent_codes = self.generator.truncate_w(latent_codes, self.truncation_psi)
        
        images = self.generate_images(latent_codes)
        
        if return_latents:
            return images, latent_codes
        return images
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.mapper.eval()
        self.text_encoder.eval()
        self.generator.eval()


# ============================================================================
# INFERENCE HELPER
# ============================================================================

def load_config_from_checkpoint(checkpoint_path: Path) -> dict:
    """Load config from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint doesn't contain config. Please provide config file.")
    return checkpoint['config']


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # Tensor is [3, H, W] in range [-1, 1]
    tensor = tensor.squeeze(0) if tensor.dim() == 4 else tensor
    
    # Convert to [0, 255]
    img_array = ((tensor.cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Convert to HWC format
    img_array = np.transpose(img_array, (1, 2, 0))
    
    return Image.fromarray(img_array)


class T2FInference:
    """Helper class for T2F inference."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize T2F model for inference.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to use ("cuda" or "cpu")
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        
        print(f"Loading T2F model from: {self.checkpoint_path}")
        
        # Load config from checkpoint
        config = load_config_from_checkpoint(self.checkpoint_path)
        config['device'] = device
        
        # Create model
        self.model = T2FModel(config=config)
        
        # Load weights
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        self.model.mapper.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval_mode()
        
        print(f"✓ Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    
    @torch.no_grad()
    def generate(self, caption: str) -> Image.Image:
        """
        Generate face image from caption.
        
        Args:
            caption: Text description of face
            
        Returns:
            Generated PIL Image
        """
        # Generate image
        image_tensor = self.model(caption)
        
        # Convert to PIL
        return tensor_to_pil(image_tensor)
    
    @torch.no_grad()
    def generate_batch(self, captions: List[str]) -> List[Image.Image]:
        """Generate multiple images from captions."""
        image_tensors = self.model(captions)
        
        images = []
        for i in range(image_tensors.shape[0]):
            images.append(tensor_to_pil(image_tensors[i]))
        
        return images
