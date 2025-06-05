# Eri - SD Forms: Beautiful AI Image & Video Generation

**⚠️ SEMI FUNCTIONAL WORK IN DEVELOPMENT ⚠️**

![Demo Video](https://user-images.githubusercontent.com/your-username/demo-video.mp4)

A beautiful, component-based visual workflow system for Stable Diffusion image and video generation with a stunning purple macOS-themed interface.

## 🎨 Features

- **Beautiful Purple macOS Theme**: Elegant LiteGraph.js interface with glass effects and smooth animations
- **Component-Based Workflow**: Drag-and-drop visual programming for AI generation
- **Multiple Model Support**: Flux, SD 1.5, SD 3.5, SDXL, Chroma, OmniGen
- **Video Generation**: AnimateDiff, LTX Video, WAN-VACE support
- **Memory Optimized**: Automatic CPU offload for large models
- **Auto Model Detection**: JSON-based configuration system

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/CodeAlexx/Eri.git
cd Eri
```

2. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers>=0.30.0 transformers>=4.38.0 accelerate
pip install fastapi uvicorn pillow numpy
pip install safetensors huggingface_hub
```

3. **Configure model paths**
Edit `model_paths_config.json` to point to your model directories:
```json
{
  "model_directories": {
    "flux": ["/path/to/your/models"],
    "sd15": ["/path/to/your/models"],
    "sdxl": ["/path/to/your/models"]
  }
}
```

4. **Start the server**
```bash
cd backend
python server.py
```

5. **Open web interface**
Navigate to: `http://localhost:8001`

## 🎯 Generate Your First Image

1. **Clear Canvas**: Start with a fresh workspace

2. **Add SDXL Model Component**: 
   - Drag an **SDXL** component from the right panel (currently the most stable option)
   
3. **Generate Basic Workflow**: 
   - Click the **Generate Basic Workflow** button to automatically add and connect required components

4. **Adjust Settings**: 
   - Modify steps (20-30 recommended)
   - Adjust CFG scale (7-8 works well)
   - Enter your prompt in the model component

5. **Generate**: Click the **Generate** button to create your first image

## 📁 Project Structure

```
Eri/
├── sd_forms/              # Core component system (THE GOLD!)
│   ├── components/        # Image/video generation components
│   ├── core/             # Pipeline system & registries
│   └── utils/            # Model path configuration
├── backend/              # FastAPI server
├── web/                  # LiteGraph.js frontend
├── model_paths_config.json # Model directory configuration
└── README.md
```

## 🔧 Key Components

### Image Generation
- **Flux Model**: Fast 4-step generation
- **SD 1.5**: Classic Stable Diffusion
- **SDXL**: High-resolution 1024x1024
- **Chroma**: Flux-based model
- **SD 3.5**: Latest Stability AI model

### Video Generation  
- **AnimateDiff**: SD 1.5 animation
- **LTX Video**: Direct video generation
- **WAN-VACE**: Video generation model

### Processing
- **Sampler**: Core generation component
- **Upscaler**: Image enhancement
- **ControlNet**: Guided generation
- **VAE**: Custom VAE loading

## 🎨 Beautiful Interface

The LiteGraph.js interface features:
- Purple gradient theme with glass morphism
- Smooth animations and hover effects
- macOS-style components and typography
- Draggable property panels
- Auto-connecting workflow nodes

## ⚙️ Configuration

### Memory Optimization
Large models automatically use CPU offload. Disable in component properties if you have enough VRAM.

### Model Auto-Detection
The system automatically finds models in:
- SwarmUI model directories
- Custom paths in `model_paths_config.json`
- HuggingFace cache

## 🐛 Known Issues

- This is development software - expect bugs!
- Large models require significant RAM/VRAM
- Some components may need additional testing
- Error handling is still being improved

## 🤝 Contributing

This is an active development project. Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share your workflows

## 📄 License

Open source - use at your own risk during development phase.

---

**Generate beautiful images and videos with the power of visual programming!**

*Built with love for the AI art community* 💜