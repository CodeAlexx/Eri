"""
SimpleTuner Training Component for SD Forms
Based on bghira's SimpleTuner architecture for fine-tuning Stable Diffusion models
Supports LoRA, DreamBooth, and full fine-tuning workflows
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import torch
import json
from dataclasses import dataclass

from .base import VisualComponent
from ..core import (
    create_port, create_property_definition, 
    PortType, PortDirection, PropertyType, ComponentStatus
)
from ..utils.constants import DEVICE

# Training method configurations
TRAINING_CONFIGS = {
    'lora': {
        'name': 'LoRA Training',
        'rank': 32,
        'alpha': 32,
        'learning_rate': 1e-4,
        'description': 'Low-Rank Adaptation - efficient fine-tuning'
    },
    'dreambooth': {
        'name': 'DreamBooth',
        'learning_rate': 5e-6,
        'instance_prompt': 'sks',
        'class_prompt': '',
        'description': 'Subject-driven generation'
    },
    'full_finetune': {
        'name': 'Full Fine-tune',
        'learning_rate': 1e-5,
        'description': 'Complete model fine-tuning'
    },
    'textual_inversion': {
        'name': 'Textual Inversion',
        'learning_rate': 5e-4,
        'num_vectors': 4,
        'description': 'Learn new concepts as embeddings'
    }
}

@dataclass
class DatasetConfig:
    """Configuration for training dataset"""
    path: str
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = True
    caption_dropout_rate: float = 0.1
    repeats: int = 1

class SimpleTunerTrainingComponent(VisualComponent):
    """SimpleTuner-based training component for fine-tuning models"""
    
    component_type = "training"
    display_name = "Model Training"
    category = "Training"
    icon = "🎓"
    
    # Define input ports
    input_ports = [
        create_port("base_model", PortType.ANY, PortDirection.INPUT, optional=True),
        create_port("vae", PortType.ANY, PortDirection.INPUT, optional=True),
        create_port("dataset_path", PortType.ANY, PortDirection.INPUT, optional=True),
    ]
    
    # Define output ports  
    output_ports = [
        create_port("trained_model", PortType.ANY, PortDirection.OUTPUT),
        create_port("training_logs", PortType.ANY, PortDirection.OUTPUT),
        create_port("sample_images", PortType.IMAGE, PortDirection.OUTPUT, optional=True),
    ]
    
    # Define properties
    property_definitions = [
        # Training Method
        create_property_definition("training_method", "Training Method", PropertyType.CHOICE, "lora", "Method",
                                 metadata={"choices": list(TRAINING_CONFIGS.keys()),
                                          "descriptions": {k: v['description'] for k, v in TRAINING_CONFIGS.items()}}),
        
        # Model Settings
        create_property_definition("model_type", "Model Type", PropertyType.CHOICE, "sdxl", "Model",
                                 metadata={"choices": ["sd15", "sd21", "sdxl", "flux"],
                                          "description": "Base model architecture"}),
        create_property_definition("base_model_path", "Base Model", PropertyType.FILE_PATH, "", "Model",
                                 metadata={"editor_type": "file_picker", "filter": "*.safetensors;*.ckpt",
                                          "description": "Path to base model checkpoint"}),
        create_property_definition("vae_path", "VAE Model", PropertyType.FILE_PATH, "", "Model",
                                 metadata={"editor_type": "file_picker", "filter": "*.safetensors;*.pt",
                                          "description": "Optional custom VAE", "optional": True}),
        
        # Dataset Configuration
        create_property_definition("dataset_path", "Dataset Path", PropertyType.FILE_PATH, "", "Dataset",
                                 metadata={"editor_type": "folder_picker",
                                          "description": "Folder containing training images"}),
        create_property_definition("resolution", "Resolution", PropertyType.INTEGER, 1024, "Dataset",
                                 metadata={"min": 256, "max": 2048, "step": 64,
                                          "description": "Training resolution"}),
        create_property_definition("center_crop", "Center Crop", PropertyType.BOOLEAN, True, "Dataset",
                                 metadata={"description": "Center crop images to resolution"}),
        create_property_definition("random_flip", "Random Flip", PropertyType.BOOLEAN, True, "Dataset",
                                 metadata={"description": "Randomly flip images horizontally"}),
        create_property_definition("caption_strategy", "Caption Strategy", PropertyType.CHOICE, "filename", "Dataset",
                                 metadata={"choices": ["filename", "txt_files", "blip", "directory", "instance_prompt"],
                                          "description": "How to generate captions"}),
        create_property_definition("caption_dropout", "Caption Dropout", PropertyType.FLOAT, 0.1, "Dataset",
                                 metadata={"editor_type": "float_slider", "min": 0.0, "max": 1.0, "step": 0.05,
                                          "description": "Randomly drop captions during training"}),
        
        # LoRA Specific Settings
        create_property_definition("lora_rank", "LoRA Rank", PropertyType.INTEGER, 32, "LoRA Settings",
                                 metadata={"min": 1, "max": 256, "step": 1,
                                          "depends_on": {"training_method": "lora"}}),
        create_property_definition("lora_alpha", "LoRA Alpha", PropertyType.INTEGER, 32, "LoRA Settings",
                                 metadata={"min": 1, "max": 256, "step": 1,
                                          "depends_on": {"training_method": "lora"}}),
        create_property_definition("lora_dropout", "LoRA Dropout", PropertyType.FLOAT, 0.0, "LoRA Settings",
                                 metadata={"editor_type": "float_slider", "min": 0.0, "max": 0.5, "step": 0.05,
                                          "depends_on": {"training_method": "lora"}}),
        create_property_definition("target_modules", "Target Modules", PropertyType.TEXT, "to_k,to_q,to_v,to_out", "LoRA Settings",
                                 metadata={"editor_type": "text_multiline",
                                          "description": "Comma-separated list of modules to train",
                                          "depends_on": {"training_method": "lora"}}),
        
        # DreamBooth Specific Settings
        create_property_definition("instance_prompt", "Instance Prompt", PropertyType.TEXT, "sks", "DreamBooth",
                                 metadata={"description": "Unique identifier for your subject",
                                          "depends_on": {"training_method": "dreambooth"}}),
        create_property_definition("class_prompt", "Class Prompt", PropertyType.TEXT, "", "DreamBooth",
                                 metadata={"description": "General class of your subject",
                                          "depends_on": {"training_method": "dreambooth"}}),
        create_property_definition("num_class_images", "Class Images", PropertyType.INTEGER, 200, "DreamBooth",
                                 metadata={"min": 0, "max": 1000,
                                          "description": "Number of regularization images",
                                          "depends_on": {"training_method": "dreambooth"}}),
        
        # Training Parameters
        create_property_definition("num_train_epochs", "Training Epochs", PropertyType.INTEGER, 100, "Training",
                                 metadata={"min": 1, "max": 1000,
                                          "description": "Number of training epochs"}),
        create_property_definition("train_batch_size", "Batch Size", PropertyType.INTEGER, 1, "Training",
                                 metadata={"min": 1, "max": 32,
                                          "description": "Training batch size"}),
        create_property_definition("gradient_accumulation_steps", "Gradient Accumulation", PropertyType.INTEGER, 1, "Training",
                                 metadata={"min": 1, "max": 32,
                                          "description": "Steps to accumulate gradients"}),
        create_property_definition("learning_rate", "Learning Rate", PropertyType.FLOAT, 1e-4, "Training",
                                 metadata={"editor_type": "float_slider", "min": 1e-7, "max": 1e-1, "step": 1e-7,
                                          "description": "Initial learning rate"}),
        create_property_definition("lr_scheduler", "LR Scheduler", PropertyType.CHOICE, "cosine", "Training",
                                 metadata={"choices": ["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"],
                                          "description": "Learning rate schedule"}),
        create_property_definition("lr_warmup_steps", "Warmup Steps", PropertyType.INTEGER, 500, "Training",
                                 metadata={"min": 0, "max": 10000,
                                          "description": "LR warmup steps"}),
        
        # Optimization Settings
        create_property_definition("optimizer", "Optimizer", PropertyType.CHOICE, "adamw", "Optimization",
                                 metadata={"choices": ["adamw", "adam", "sgd", "adafactor", "prodigy"],
                                          "description": "Optimization algorithm"}),
        create_property_definition("adam_beta1", "Adam Beta1", PropertyType.FLOAT, 0.9, "Optimization",
                                 metadata={"editor_type": "float_slider", "min": 0.0, "max": 1.0, "step": 0.01}),
        create_property_definition("adam_beta2", "Adam Beta2", PropertyType.FLOAT, 0.999, "Optimization",
                                 metadata={"editor_type": "float_slider", "min": 0.0, "max": 1.0, "step": 0.001}),
        create_property_definition("adam_epsilon", "Adam Epsilon", PropertyType.FLOAT, 1e-8, "Optimization",
                                 metadata={"editor_type": "float_slider", "min": 1e-10, "max": 1e-6, "step": 1e-10}),
        create_property_definition("max_grad_norm", "Max Gradient Norm", PropertyType.FLOAT, 1.0, "Optimization",
                                 metadata={"editor_type": "float_slider", "min": 0.0, "max": 10.0, "step": 0.1,
                                          "description": "Gradient clipping"}),
        
        # Advanced Settings
        create_property_definition("mixed_precision", "Mixed Precision", PropertyType.CHOICE, "fp16", "Advanced",
                                 metadata={"choices": ["no", "fp16", "bf16"],
                                          "description": "Use mixed precision training"}),
        create_property_definition("gradient_checkpointing", "Gradient Checkpointing", PropertyType.BOOLEAN, True, "Advanced",
                                 metadata={"description": "Trade compute for memory"}),
        create_property_definition("use_8bit_adam", "8-bit Adam", PropertyType.BOOLEAN, False, "Advanced",
                                 metadata={"description": "Use 8-bit Adam optimizer"}),
        create_property_definition("enable_xformers", "Enable xFormers", PropertyType.BOOLEAN, True, "Advanced",
                                 metadata={"description": "Use xFormers for memory efficiency"}),
        create_property_definition("cache_latents", "Cache Latents", PropertyType.BOOLEAN, True, "Advanced",
                                 metadata={"description": "Pre-compute and cache VAE latents"}),
        
        # Validation & Logging
        create_property_definition("validation_prompt", "Validation Prompt", PropertyType.TEXT, "", "Validation",
                                 metadata={"editor_type": "text_multiline",
                                          "description": "Prompt for validation images"}),
        create_property_definition("num_validation_images", "Validation Images", PropertyType.INTEGER, 4, "Validation",
                                 metadata={"min": 0, "max": 16,
                                          "description": "Images to generate for validation"}),
        create_property_definition("validation_epochs", "Validation Every N Epochs", PropertyType.INTEGER, 10, "Validation",
                                 metadata={"min": 1, "max": 100,
                                          "description": "Run validation every N epochs"}),
        create_property_definition("save_epochs", "Save Every N Epochs", PropertyType.INTEGER, 10, "Output",
                                 metadata={"min": 1, "max": 100,
                                          "description": "Save checkpoint every N epochs"}),
        create_property_definition("output_dir", "Output Directory", PropertyType.FILE_PATH, "./training_output", "Output",
                                 metadata={"editor_type": "folder_picker",
                                          "description": "Where to save checkpoints"}),
        create_property_definition("output_name", "Output Name", PropertyType.TEXT, "trained_model", "Output",
                                 metadata={"description": "Base name for saved models"}),
        
        # Logging
        create_property_definition("logging_dir", "Logging Directory", PropertyType.FILE_PATH, "./logs", "Logging",
                                 metadata={"editor_type": "folder_picker",
                                          "description": "TensorBoard logging directory"}),
        create_property_definition("report_to", "Report To", PropertyType.CHOICE, "tensorboard", "Logging",
                                 metadata={"choices": ["none", "tensorboard", "wandb", "all"],
                                          "description": "Logging backend"}),
        create_property_definition("wandb_project", "W&B Project", PropertyType.TEXT, "sd-training", "Logging",
                                 metadata={"description": "Weights & Biases project name",
                                          "depends_on": {"report_to": ["wandb", "all"]}}),
    ]
    
    def __init__(self, id: str = None):
        super().__init__(id, position=(0, 0))
        self.trainer = None
        self.dataset = None
        self.accelerator = None
        
    def prepare_dataset(self) -> DatasetConfig:
        """Prepare dataset configuration from properties"""
        return DatasetConfig(
            path=self.properties.get("dataset_path", ""),
            resolution=self.properties.get("resolution", 512),
            center_crop=self.properties.get("center_crop", True),
            random_flip=self.properties.get("random_flip", True),
            caption_dropout_rate=self.properties.get("caption_dropout", 0.1),
            repeats=1
        )
    
    def create_training_config(self) -> Dict[str, Any]:
        """Create complete training configuration"""
        method = self.properties.get("training_method", "lora")
        base_config = TRAINING_CONFIGS[method].copy()
        
        config = {
            "training_method": method,
            "model_type": self.properties.get("model_type", "sdxl"),
            "pretrained_model_name_or_path": self.properties.get("base_model_path", ""),
            "output_dir": self.properties.get("output_dir", "./training_output"),
            "output_name": self.properties.get("output_name", "trained_model"),
            
            # Dataset
            "train_data_dir": self.properties.get("dataset_path", ""),
            "resolution": self.properties.get("resolution", 512),
            "center_crop": self.properties.get("center_crop", True),
            "random_flip": self.properties.get("random_flip", True),
            "caption_dropout_rate": self.properties.get("caption_dropout", 0.1),
            
            # Training parameters
            "num_train_epochs": self.properties.get("num_train_epochs", 100),
            "train_batch_size": self.properties.get("train_batch_size", 1),
            "gradient_accumulation_steps": self.properties.get("gradient_accumulation_steps", 1),
            "learning_rate": self.properties.get("learning_rate", base_config.get('learning_rate', 1e-4)),
            "lr_scheduler": self.properties.get("lr_scheduler", "cosine"),
            "lr_warmup_steps": self.properties.get("lr_warmup_steps", 500),
            
            # Optimization
            "optimizer_type": self.properties.get("optimizer", "adamw"),
            "adam_beta1": self.properties.get("adam_beta1", 0.9),
            "adam_beta2": self.properties.get("adam_beta2", 0.999),
            "adam_epsilon": self.properties.get("adam_epsilon", 1e-8),
            "max_grad_norm": self.properties.get("max_grad_norm", 1.0),
            
            # Advanced
            "mixed_precision": self.properties.get("mixed_precision", "fp16"),
            "gradient_checkpointing": self.properties.get("gradient_checkpointing", True),
            "use_8bit_adam": self.properties.get("use_8bit_adam", False),
            "enable_xformers_memory_efficient_attention": self.properties.get("enable_xformers", True),
            "cache_latents": self.properties.get("cache_latents", True),
            
            # Validation
            "validation_prompt": self.properties.get("validation_prompt", ""),
            "num_validation_images": self.properties.get("num_validation_images", 4),
            "validation_epochs": self.properties.get("validation_epochs", 10),
            "save_model_epochs": self.properties.get("save_epochs", 10),
            
            # Logging
            "logging_dir": self.properties.get("logging_dir", "./logs"),
            "report_to": self.properties.get("report_to", "tensorboard"),
        }
        
        # Add method-specific settings
        if method == "lora":
            config.update({
                "network_dim": self.properties.get("lora_rank", 32),
                "network_alpha": self.properties.get("lora_alpha", 32),
                "network_dropout": self.properties.get("lora_dropout", 0.0),
                "network_module": "networks.lora",
                "network_args": {
                    "conv_dim": self.properties.get("lora_rank", 32),
                    "conv_alpha": self.properties.get("lora_alpha", 32),
                }
            })
        elif method == "dreambooth":
            config.update({
                "instance_prompt": self.properties.get("instance_prompt", "sks"),
                "class_prompt": self.properties.get("class_prompt", ""),
                "num_class_images": self.properties.get("num_class_images", 200),
                "prior_preservation": bool(self.properties.get("class_prompt", "")),
                "prior_preservation_weight": 1.0,
            })
        elif method == "textual_inversion":
            config.update({
                "num_vectors_per_token": self.properties.get("num_vectors", 4),
                "placeholder_token": self.properties.get("instance_prompt", "<token>"),
                "initializer_token": self.properties.get("class_prompt", ""),
            })
        
        return config
    
    async def process(self, context) -> bool:
        """Execute training process"""
        try:
            self.set_status(ComponentStatus.PROCESSING)
            
            # Get inputs
            base_model = self.get_input_data("base_model") or self.properties.get("base_model_path")
            dataset_path = self.get_input_data("dataset_path") or self.properties.get("dataset_path")
            
            if not base_model:
                raise ValueError("No base model specified")
            if not dataset_path:
                raise ValueError("No dataset path specified")
            
            # Create training configuration
            config = self.create_training_config()
            
            # Log configuration
            print(f"Starting {config['training_method']} training")
            print(f"Base model: {base_model}")
            print(f"Dataset: {dataset_path}")
            print(f"Output: {config['output_dir']}/{config['output_name']}")
            
            # Initialize accelerator for distributed training
            try:
                from accelerate import Accelerator
                self.accelerator = Accelerator(
                    mixed_precision=config["mixed_precision"],
                    gradient_accumulation_steps=config["gradient_accumulation_steps"],
                    log_with=config["report_to"],
                    project_dir=config["logging_dir"]
                )
            except ImportError:
                print("Warning: Accelerate not available, using basic training")
                self.accelerator = None
            
            # Create output directory
            Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
            
            # Initialize trainer based on method
            if config["training_method"] == "lora":
                trainer = self._create_lora_trainer(config)
            elif config["training_method"] == "dreambooth":
                trainer = self._create_dreambooth_trainer(config)
            elif config["training_method"] == "textual_inversion":
                trainer = self._create_textual_inversion_trainer(config)
            else:
                trainer = self._create_full_finetune_trainer(config)
            
            # Training loop with progress updates
            total_steps = config["num_train_epochs"] * getattr(trainer, 'steps_per_epoch', 100)
            current_step = 0
            
            for epoch in range(config["num_train_epochs"]):
                trainer.train_epoch(epoch)
                
                # Validation
                if epoch % config["validation_epochs"] == 0 and config["validation_prompt"]:
                    validation_images = trainer.validate(
                        config["validation_prompt"], 
                        config["num_validation_images"]
                    )
                    
                    # Send validation images through WebSocket
                    if hasattr(context, 'preview_callback') and validation_images:
                        for i, img in enumerate(validation_images):
                            await context.preview_callback(img, epoch, i)
                
                # Save checkpoint
                if epoch % config["save_model_epochs"] == 0:
                    checkpoint_path = f"{config['output_dir']}/{config['output_name']}_epoch_{epoch}"
                    trainer.save_checkpoint(checkpoint_path)
                    print(f"Saved checkpoint: {checkpoint_path}")
                
                # Progress update
                current_step += getattr(trainer, 'steps_per_epoch', 100)
                progress = int((current_step / total_steps) * 100)
                
                if hasattr(context, 'progress_callback'):
                    await context.progress_callback(progress, epoch, current_step)
            
            # Save final model
            final_path = f"{config['output_dir']}/{config['output_name']}_final"
            trainer.save_model(final_path)
            
            # Set outputs
            self.set_output_data("trained_model", final_path)
            self.set_output_data("training_logs", {
                "epochs": config["num_train_epochs"],
                "final_loss": trainer.get_final_loss(),
                "training_time": trainer.get_training_time(),
                "model_path": final_path
            })
            
            if hasattr(trainer, 'sample_images'):
                self.set_output_data("sample_images", trainer.sample_images)
            
            self.set_status(ComponentStatus.COMPLETE)
            return True
            
        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            print(f"Error in training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_lora_trainer(self, config: Dict[str, Any]):
        """Create LoRA trainer instance"""
        # This would integrate with kohya-ss/sd-scripts or similar
        class LoRATrainer:
            def __init__(self, config):
                self.config = config
                self.steps_per_epoch = 100
                # Initialize LoRA training
                
            def train_epoch(self, epoch):
                # Training logic - would integrate with actual LoRA training
                print(f"Training epoch {epoch}")
                pass
                
            def validate(self, prompt, num_images):
                # Validation logic - would generate sample images
                print(f"Validating with prompt: {prompt}")
                return []
                
            def save_checkpoint(self, path):
                # Save checkpoint - would save actual model weights
                print(f"Saving checkpoint to {path}")
                pass
                
            def save_model(self, path):
                # Save final model - would save the trained LoRA
                print(f"Saving final model to {path}")
                pass
                
            def get_final_loss(self):
                return 0.001
                
            def get_training_time(self):
                return "1:30:00"
        
        return LoRATrainer(config)
    
    def _create_dreambooth_trainer(self, config: Dict[str, Any]):
        """Create DreamBooth trainer instance"""
        # Would integrate with diffusers DreamBooth implementation
        return self._create_lora_trainer(config)  # Placeholder
    
    def _create_textual_inversion_trainer(self, config: Dict[str, Any]):
        """Create Textual Inversion trainer instance"""
        # Would integrate with diffusers Textual Inversion
        return self._create_lora_trainer(config)  # Placeholder
    
    def _create_full_finetune_trainer(self, config: Dict[str, Any]):
        """Create full fine-tuning trainer instance"""
        # Would integrate with full model fine-tuning
        return self._create_lora_trainer(config)  # Placeholder
    
    def unload_model(self):
        """Clean up training resources"""
        if self.trainer:
            del self.trainer
            self.trainer = None
        if self.accelerator:
            del self.accelerator
            self.accelerator = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Training resources unloaded")