"""
◈ PARALLAX STUDIO v1.2
Transform photos into living, breathing displays

Your Photos. Alive.
"""

import streamlit as st
import numpy as np
from pathlib import Path
import tempfile
import time
import subprocess
import shutil
import os
from PIL import Image
import io
import torch

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Parallax Studio v1.2",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "sharp_checkpoint": "sharp_2572gikvuh.pt",
    "sharp_cache": Path.home() / ".cache" / "sharp",
    "sharp_repo": "apple/Sharp",
    "qwen_model": "Qwen/Qwen-Image-Edit",
    "version": "1.2.0",
}

ASPECT_RATIOS = {
    "16:9 (Standard Widescreen)": {
        "ratio": 16/9,
        "resolutions": [
            ("4K UHD", 3840, 2160),
            ("2K QHD", 2560, 1440),
            ("1080p Full HD", 1920, 1080),
        ]
    },
    "16:10 (Common Monitor)": {
        "ratio": 16/10,
        "resolutions": [
            ("WQXGA", 2560, 1600),
            ("WUXGA", 1920, 1200),
        ]
    },
    "21:9 (Ultrawide)": {
        "ratio": 21/9,
        "resolutions": [
            ("UWQHD", 3440, 1440),
            ("UW-UXGA", 2560, 1080),
        ]
    },
    "32:9 (Super Ultrawide / Video Wall)": {
        "ratio": 32/9,
        "resolutions": [
            ("Dual 4K", 7680, 2160),
            ("Dual QHD", 5120, 1440),
            ("Dual FHD", 3840, 1080),
        ]
    },
    "Custom": {
        "ratio": None,
        "resolutions": []
    }
}

STYLE_PRESETS = {
    "None (Keep Original)": "",
    "Studio Ghibli": "Transform this image into Studio Ghibli anime style with soft colors and whimsical atmosphere",
    "Oil Painting": "Transform this image into a classical oil painting with visible brushstrokes and rich colors",
    "Watercolor": "Transform this image into a delicate watercolor painting with soft edges and translucent colors",
    "Cyberpunk": "Transform this image into cyberpunk aesthetic with neon lights and futuristic elements",
    "Golden Hour": "Transform this image to have warm golden hour lighting with dramatic long shadows",
    "Dramatic Sky": "Replace the sky with a dramatic sunset with vibrant orange and purple colors",
    "Impressionist": "Transform this image into impressionist painting style like Monet",
    "Add Atmospheric Fog": "Add atmospheric fog and mist throughout the scene for depth and mystery",
    "Noir": "Transform this image into black and white film noir style with dramatic shadows",
    "Vintage Film": "Transform this image to look like a vintage 1970s film photograph with warm tones",
    "Winter Scene": "Transform this scene into winter with snow covering all surfaces",
}

# ============================================================================
# CSS STYLING
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-tertiary: #1a1a25;
        --bg-card: linear-gradient(145deg, #151520 0%, #0d0d14 100%);
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-tertiary: #a855f7;
        --accent-glow: rgba(99, 102, 241, 0.4);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-accent: rgba(99, 102, 241, 0.3);
        --success: #10b981;
        --warning: #f59e0b;
    }
    
    .stApp {
        background: var(--bg-primary);
        background-image: 
            radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.06) 0%, transparent 50%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .main .block-container {
        max-width: 1400px;
        padding: 2rem 3rem 4rem;
    }
    
    .hero-container {
        text-align: center;
        padding: 2rem 0 3rem;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid var(--border-accent);
        border-radius: 100px;
        padding: 0.5rem 1.25rem;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--accent-primary);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 50%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    
    .hero-tagline {
        font-size: 1.5rem;
        color: var(--accent-secondary);
        font-weight: 300;
        font-style: italic;
        margin: 0 0 1rem 0;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    .studio-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .studio-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.4) 50%, transparent 100%);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .card-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 8px 32px var(--accent-glow);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .card-description {
        font-size: 0.9rem;
        color: var(--text-muted);
        margin: 0;
    }
    
    .step-indicator {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        margin-right: 0.75rem;
    }
    
    .optional-badge {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-left: 0.5rem;
    }
    
    .stSlider > div > div {background: var(--bg-tertiary) !important;}
    .stSlider > div > div > div > div {background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)) !important;}
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 24px var(--accent-glow) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px var(--accent-glow) !important;
    }
    
    .stProgress > div > div {background: var(--bg-tertiary) !important; border-radius: 10px !important;}
    .stProgress > div > div > div {background: linear-gradient(90deg, var(--accent-primary), var(--accent-tertiary)) !important; border-radius: 10px !important;}
    
    .metric-container {
        background: var(--bg-tertiary);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid var(--border-subtle);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 100px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-ready {
        background: rgba(16, 185, 129, 0.15);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-pending {
        background: rgba(100, 116, 139, 0.15);
        color: var(--text-muted);
        border: 1px solid var(--border-subtle);
    }
    
    .status-skipped {
        background: rgba(245, 158, 11, 0.15);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .preview-container {
        background: var(--bg-secondary);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid var(--border-subtle);
    }
    
    .preview-container img, .preview-container video {
        border-radius: 8px;
        width: 100%;
    }
    
    .param-group {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .param-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .param-hint {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    
    .gpu-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: var(--success);
    }
    
    .success-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .streamlit-expanderHeader {
        background: var(--bg-tertiary) !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.5rem !important;
    }
    
    .studio-footer {
        text-align: center;
        padding: 3rem 0 1rem;
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    
    .footer-logo {
        font-weight: 600;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }
    p, span, label { color: var(--text-secondary) !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================

defaults = {
    'source_image': None,
    'enhanced_image': None,
    'processed_image': None,
    'gaussian_path': None,
    'video_path': None,
    'work_dir': None,
    'target_width': 5120,
    'target_height': 1440,
    'target_ratio': 32/9,
    'use_qwen': False,
    'qwen_pipeline': None,
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_status_badge(status):
    icons = {'ready': '✓', 'pending': '○', 'skipped': '—'}
    return f'<span class="status-badge status-{status}">{icons.get(status, "○")} {status.title()}</span>'

def check_gpu():
    try:
        return torch.cuda.is_available()
    except:
        return False

def get_gpu_name():
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except:
        pass
    return "Not detected"

def setup_work_dir():
    if st.session_state.work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="parallax_"))
        (work_dir / "input").mkdir()
        (work_dir / "enhanced").mkdir()
        (work_dir / "processed").mkdir()
        (work_dir / "gaussians").mkdir()
        (work_dir / "frames").mkdir()
        st.session_state.work_dir = work_dir
    return st.session_state.work_dir

def ensure_sharp_checkpoint():
    checkpoint_path = CONFIG["sharp_cache"] / CONFIG["sharp_checkpoint"]
    if checkpoint_path.exists():
        return checkpoint_path
    CONFIG["sharp_cache"].mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=CONFIG["sharp_repo"],
        filename=CONFIG["sharp_checkpoint"],
        local_dir=CONFIG["sharp_cache"]
    ))

def load_qwen_pipeline():
    if st.session_state.qwen_pipeline is None:
        from diffusers import QwenImageEditPipeline
        pipeline = QwenImageEditPipeline.from_pretrained(CONFIG["qwen_model"])
        pipeline.to(torch.bfloat16)
        pipeline.to("cuda")
        st.session_state.qwen_pipeline = pipeline
    return st.session_state.qwen_pipeline

def apply_qwen_enhancement(image, prompt, cfg_scale=4.0, steps=50, seed=0):
    pipeline = load_qwen_pipeline()
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": steps,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
    return output.images[0]

def process_image_to_aspect(image, target_ratio, target_width, target_height):
    width, height = image.size
    current_ratio = width / height
    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        image = image.crop((left, 0, left + new_width, height))
    else:
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        image = image.crop((0, top, width, top + new_height))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

def run_sharp_predict(input_dir, output_dir, checkpoint_path):
    cmd = ["sharp", "predict", "-i", str(input_dir), "-o", str(output_dir), "-c", str(checkpoint_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout + result.stderr

def render_parallax_video(gaussian_path, output_dir, width, height, frames, amplitude, fps):
    """Render parallax video from Gaussian splat."""
    from plyfile import PlyData
    from gsplat import rasterization
    
    # Load Gaussian splat
    plydata = PlyData.read(str(gaussian_path))
    v = plydata['vertex']
    
    device = "cuda"
    means = torch.tensor(np.stack([v['x'], v['y'], v['z']], axis=-1), dtype=torch.float32, device=device)
    scales = torch.exp(torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1), dtype=torch.float32, device=device))
    quats = torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1), dtype=torch.float32, device=device)
    colors = torch.tensor(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1), dtype=torch.float32, device=device)
    opacities = torch.sigmoid(torch.tensor(np.array(v['opacity']), dtype=torch.float32, device=device))
    
    fx = fy = width * 0.8
    K = torch.tensor([[fx, 0, width/2], [0, fy, height/2], [0, 0, 1]], device=device)
    scene_z = float(means[:, 2].mean())
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(frames):
        t = 2 * np.pi * i / frames
        x_offset = amplitude * np.sin(t)
        cam_pos = np.array([x_offset, 0.0, scene_z - 2.5])
        
        R = np.eye(3)
        T = -R @ cam_pos
        viewmat = np.eye(4)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = T
        viewmat = torch.tensor(viewmat, dtype=torch.float32, device=device)
        
        rendered, _, _ = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmat[None], Ks=K[None],
            width=width, height=height
        )
        
        frame = (rendered[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(frame).save(output_dir / f"frame_{i:05d}.png")
        yield i + 1, frames
    
    # Encode video
    video_path = output_dir.parent / "parallax_output.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(output_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        str(video_path)
    ], check=True, capture_output=True)
    
    yield video_path, None


# ============================================================================
# HEADER
# ============================================================================

st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge">◈ Version {CONFIG['version']} — Qwen + SHARP Pipeline</div>
    <h1 class="hero-title">Parallax Studio</h1>
    <p class="hero-tagline">Your Photos. Alive.</p>
    <p class="hero-subtitle">Transform any photograph into a living, breathing display with real depth and dimension</p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# COLLAPSIBLE INFO SECTIONS
# ============================================================================

with st.expander("📖 How to Use This App", expanded=False):
    st.markdown("""
    ### Welcome to Parallax Studio!
    
    This app transforms a regular photograph into a video with **depth and dimension** — making it look like you're peering through a window into the scene.
    
    ---
    
    #### The Simple Version
    
    1. **Upload** a photo
    2. **Optionally** transform it into art (Ghibli, oil painting, etc.)
    3. **Choose** your display size
    4. **Click** a few buttons
    5. **Download** a video that makes people say "whoa"
    
    ---
    
    #### Step by Step
    
    **Step 1: Upload Your Image**
    Drag and drop or click to upload. Higher resolution = better results.
    
    **Step 2: Enhance (Optional)**
    Want your photo to look like a Studio Ghibli scene? An oil painting? Toggle enhancement ON, pick a style, and watch the magic. Skip this if your photo is already perfect.
    
    **Step 3: Choose Format**
    Pick the aspect ratio that matches your display:
    - **16:9** — Regular TVs and monitors
    - **21:9** — Ultrawide monitors  
    - **32:9** — Video walls and super ultrawides
    
    **Step 4: Generate 3D**
    One click. The AI figures out what's close, what's far, and builds an invisible 3D model.
    
    **Step 5: Render Video**
    Adjust how dramatic the depth effect is, hit render, and wait a few minutes.
    
    **Step 6: Download & Display**
    Put it on your video wall. Watch people stop and stare.
    
    ---
    
    #### Tips for Best Results
    
    ✓ **Scenes with depth work best** — landscapes, cityscapes, interiors
    ✓ **Higher resolution = better quality**
    ✓ **Start with subtle settings** — you can always redo with more drama
    ✗ **Avoid flat subjects** — documents, walls, close-up faces
    """)

with st.expander("ℹ️ About Parallax Studio", expanded=False):
    st.markdown("""
    ### The Technology
    
    **Qwen-Image-Edit** (Optional Enhancement)
    A 20-billion parameter AI from Alibaba that can transform your photo into any artistic style while preserving its structure. One photo becomes infinite possibilities.
    
    **Apple SHARP**
    Cutting-edge research from Apple that looks at a single photo and understands its 3D structure. In under a second, it creates a complete 3D representation using "Gaussian Splatting."
    
    **The Parallax Effect**
    When you move your head in real life, nearby things shift more than distant things. That's motion parallax — one of the strongest depth cues your brain uses. The rendered video simulates a camera gently drifting, and because we know the 3D structure, objects move at different speeds based on their depth. Your brain sees a flat screen but perceives real space.
    
    ---
    
    ### Why It Works
    
    People will stop and stare at your display because their brain is receiving depth signals it doesn't expect from a flat screen. It's not quite 3D, but it *feels* more real than any static image.
    
    ---
    
    ### Credits
    
    - **SHARP**: Apple ML Research (arXiv:2512.10685)
    - **Qwen-Image-Edit**: Alibaba Qwen Team
    - **Built with**: Streamlit, PyTorch, gsplat, FFmpeg
    """)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# GPU STATUS
# ============================================================================

gpu_available = check_gpu()
gpu_name = get_gpu_name()

col_gpu = st.columns([1, 2, 1])[1]
with col_gpu:
    if gpu_available:
        st.markdown(f'<div style="text-align:center;margin-bottom:2rem;"><span class="gpu-indicator">⚡ {gpu_name}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;margin-bottom:2rem;"><span class="status-badge status-pending">⚠ No CUDA GPU — rendering will fail</span></div>', unsafe_allow_html=True)


# ============================================================================
# STATUS OVERVIEW
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    status = 'ready' if st.session_state.source_image else 'pending'
    st.markdown(f'<div class="metric-container">{get_status_badge(status)}<div style="margin-top:0.5rem;font-size:0.75rem;color:#64748b;">SOURCE</div></div>', unsafe_allow_html=True)

with col2:
    if st.session_state.use_qwen:
        status = 'ready' if st.session_state.enhanced_image else 'pending'
    else:
        status = 'skipped'
    st.markdown(f'<div class="metric-container">{get_status_badge(status)}<div style="margin-top:0.5rem;font-size:0.75rem;color:#64748b;">ENHANCED</div></div>', unsafe_allow_html=True)

with col3:
    status = 'ready' if st.session_state.processed_image else 'pending'
    st.markdown(f'<div class="metric-container">{get_status_badge(status)}<div style="margin-top:0.5rem;font-size:0.75rem;color:#64748b;">FORMATTED</div></div>', unsafe_allow_html=True)

with col4:
    status = 'ready' if st.session_state.gaussian_path else 'pending'
    st.markdown(f'<div class="metric-container">{get_status_badge(status)}<div style="margin-top:0.5rem;font-size:0.75rem;color:#64748b;">3D MODEL</div></div>', unsafe_allow_html=True)

with col5:
    status = 'ready' if st.session_state.video_path else 'pending'
    st.markdown(f'<div class="metric-container">{get_status_badge(status)}<div style="margin-top:0.5rem;font-size:0.75rem;color:#64748b;">VIDEO</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# STEP 1: UPLOAD
# ============================================================================

st.markdown("""
<div class="studio-card">
    <div class="card-header">
        <div class="card-icon">📷</div>
        <div>
            <h3 class="card-title"><span class="step-indicator">1</span>Upload Your Photo</h3>
            <p class="card-description">High-resolution images produce the best results</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

upload_col1, upload_col2 = st.columns([2, 1])

with upload_col1:
    uploaded = st.file_uploader("Drop image here", type=['png', 'jpg', 'jpeg', 'tiff'], label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.session_state.source_image = img
        st.markdown('<div class="preview-container">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with upload_col2:
    if st.session_state.source_image:
        img = st.session_state.source_image
        st.markdown(f"""
        <div class="param-group">
            <div class="param-label">Image Properties</div>
            <p style="font-family:'JetBrains Mono';font-size:1.1rem;margin:0.5rem 0;">{img.size[0]} × {img.size[1]}</p>
            <p style="color:#64748b;font-size:0.85rem;margin:0;">{img.size[0]*img.size[1]/1_000_000:.1f} megapixels</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# STEP 2: ENHANCE (OPTIONAL)
# ============================================================================

if st.session_state.source_image:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="studio-card">
        <div class="card-header">
            <div class="card-icon">✨</div>
            <div>
                <h3 class="card-title"><span class="step-indicator">2</span>Enhance with AI<span class="optional-badge">Optional</span></h3>
                <p class="card-description">Transform your photo into art before adding depth</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    enh_col1, enh_col2 = st.columns([1, 1])
    
    with enh_col1:
        st.session_state.use_qwen = st.toggle("Enable Style Enhancement", value=st.session_state.use_qwen)
        
        if st.session_state.use_qwen:
            st.markdown('<div class="param-group">', unsafe_allow_html=True)
            style = st.selectbox("Style Preset", list(STYLE_PRESETS.keys()), index=1)
            
            if style == "None (Keep Original)":
                custom_prompt = st.text_area("Custom Prompt", placeholder="Describe the transformation...")
                prompt = custom_prompt
            else:
                prompt = STYLE_PRESETS[style]
                st.markdown(f'<p class="param-hint">{prompt[:100]}...</p>', unsafe_allow_html=True)
            
            cfg = st.slider("Style Strength", 1.0, 8.0, 4.0, 0.5)
            seed = st.number_input("Seed", 0, 999999, 0)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with enh_col2:
        if st.session_state.use_qwen:
            if st.button("✨ Apply Enhancement", use_container_width=True):
                if prompt:
                    with st.spinner("Loading Qwen model (~40GB on first run)..."):
                        progress = st.progress(0)
                        status = st.empty()
                        
                        status.markdown("*Loading Qwen-Image-Edit...*")
                        progress.progress(20)
                        
                        enhanced = apply_qwen_enhancement(
                            st.session_state.source_image,
                            prompt, cfg_scale=cfg, seed=seed
                        )
                        
                        progress.progress(100)
                        st.session_state.enhanced_image = enhanced
                        st.rerun()
        
        if st.session_state.enhanced_image:
            st.markdown('<div class="preview-container">', unsafe_allow_html=True)
            st.image(st.session_state.enhanced_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<p style="text-align:center;font-size:0.85rem;">✓ Enhanced</p>', unsafe_allow_html=True)


# ============================================================================
# STEP 3: FORMAT
# ============================================================================

if st.session_state.source_image:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="studio-card">
        <div class="card-header">
            <div class="card-icon">⬡</div>
            <div>
                <h3 class="card-title"><span class="step-indicator">3</span>Format for Display</h3>
                <p class="card-description">Choose aspect ratio and resolution for your screen</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    fmt_col1, fmt_col2 = st.columns([1, 1])
    
    with fmt_col1:
        aspect = st.selectbox("Aspect Ratio", list(ASPECT_RATIOS.keys()), index=3)
        
        if aspect == "Custom":
            c1, c2 = st.columns(2)
            with c1:
                target_width = st.number_input("Width", 640, 8192, 1920)
            with c2:
                target_height = st.number_input("Height", 360, 4320, 1080)
            target_ratio = target_width / target_height
        else:
            res_opts = ASPECT_RATIOS[aspect]["resolutions"]
            res_labels = [f"{n} ({w}×{h})" for n, w, h in res_opts]
            res_choice = st.selectbox("Resolution", res_labels)
            idx = res_labels.index(res_choice)
            _, target_width, target_height = res_opts[idx]
            target_ratio = ASPECT_RATIOS[aspect]["ratio"]
    
    with fmt_col2:
        # Determine which image to process
        if st.session_state.use_qwen and st.session_state.enhanced_image:
            source_for_processing = st.session_state.enhanced_image
            source_label = "enhanced image"
        else:
            source_for_processing = st.session_state.source_image
            source_label = "original image"
        
        if st.button("Crop & Resize", use_container_width=True):
            processed = process_image_to_aspect(source_for_processing, target_ratio, target_width, target_height)
            st.session_state.processed_image = processed
            st.session_state.target_width = target_width
            st.session_state.target_height = target_height
            st.rerun()
        
        if st.session_state.processed_image:
            st.markdown('<div class="preview-container">', unsafe_allow_html=True)
            st.image(st.session_state.processed_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            w, h = st.session_state.processed_image.size
            st.markdown(f'<p style="text-align:center;font-size:0.85rem;">✓ {w}×{h}</p>', unsafe_allow_html=True)


# ============================================================================
# STEP 4: 3D EXTRACTION
# ============================================================================

if st.session_state.processed_image:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="studio-card">
        <div class="card-header">
            <div class="card-icon">◇</div>
            <div>
                <h3 class="card-title"><span class="step-indicator">4</span>Extract 3D Structure</h3>
                <p class="card-description">SHARP analyzes depth and builds a 3D model</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    g_col1, g_col2 = st.columns([2, 1])
    
    with g_col2:
        if st.button("Run SHARP", use_container_width=True):
            work_dir = setup_work_dir()
            input_path = work_dir / "processed" / "frame.png"
            st.session_state.processed_image.save(input_path)
            
            status = st.empty()
            progress = st.progress(0)
            
            status.markdown("*Checking SHARP model...*")
            checkpoint = ensure_sharp_checkpoint()
            progress.progress(30)
            
            status.markdown("*Extracting 3D structure...*")
            success, output = run_sharp_predict(work_dir / "processed", work_dir / "gaussians", checkpoint)
            progress.progress(100)
            
            if success:
                plys = list((work_dir / "gaussians").glob("*.ply"))
                if plys:
                    st.session_state.gaussian_path = plys[0]
                    status.markdown("*✓ 3D model created*")
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("SHARP failed")
                st.code(output)
    
    with g_col1:
        if st.session_state.gaussian_path:
            size_mb = st.session_state.gaussian_path.stat().st_size / 1024 / 1024
            st.markdown(f"""
            <div class="param-group" style="text-align:center;padding:2rem;">
                <div style="font-size:3rem;margin-bottom:1rem;">◇</div>
                <div class="param-label">3D Gaussian Splat Ready</div>
                <p style="font-family:'JetBrains Mono';color:#64748b;margin-top:0.5rem;">{size_mb:.1f} MB</p>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# STEP 5: RENDER
# ============================================================================

if st.session_state.gaussian_path:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="studio-card">
        <div class="card-header">
            <div class="card-icon">▶</div>
            <div>
                <h3 class="card-title"><span class="step-indicator">5</span>Render Parallax Video</h3>
                <p class="card-description">Animate the camera to reveal depth</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    r_col1, r_col2 = st.columns([1, 1])
    
    with r_col1:
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        amplitude = st.slider("Depth Intensity", 0.05, 0.40, 0.15, 0.01, help="How dramatic the parallax effect is")
        duration = st.slider("Loop Duration (sec)", 5, 30, 10)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with r_col2:
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        fps = st.selectbox("Frame Rate", [24, 30, 60], index=1)
        
        full_w, full_h = st.session_state.target_width, st.session_state.target_height
        res_opt = st.selectbox("Render Resolution", [
            f"Full ({full_w}×{full_h})",
            f"Half ({full_w//2}×{full_h//2})",
            f"Preview ({full_w//4}×{full_h//4})"
        ])
        
        if "Full" in res_opt:
            rw, rh = full_w, full_h
        elif "Half" in res_opt:
            rw, rh = full_w//2, full_h//2
        else:
            rw, rh = full_w//4, full_h//4
        st.markdown('</div>', unsafe_allow_html=True)
    
    total_frames = duration * fps
    st.markdown(f'<p style="text-align:center;color:#64748b;font-size:0.9rem;">Total frames: <b>{total_frames}</b></p>', unsafe_allow_html=True)
    
    render_col = st.columns([1, 2, 1])[1]
    with render_col:
        if st.button("✦ Begin Render", use_container_width=True):
            work_dir = st.session_state.work_dir
            progress = st.progress(0)
            status = st.empty()
            
            status.markdown("*Initializing renderer...*")
            
            for result, total in render_parallax_video(
                st.session_state.gaussian_path,
                work_dir / "frames",
                rw, rh, total_frames, amplitude, fps
            ):
                if total is None:
                    st.session_state.video_path = Path(result)
                    status.markdown("*✓ Render complete*")
                    time.sleep(1)
                    st.rerun()
                else:
                    progress.progress(result / total)
                    status.markdown(f"*Rendering frame {result}/{total}...*")


# ============================================================================
# STEP 6: OUTPUT
# ============================================================================

if st.session_state.video_path and st.session_state.video_path.exists():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="studio-card">
        <div class="card-header">
            <div class="card-icon">✦</div>
            <div>
                <h3 class="card-title"><span class="step-indicator">6</span>Your Video is Ready</h3>
                <p class="card-description">Preview and download</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-banner">
        <span style="font-size:2rem;">✦</span>
        <h4 style="color:#10b981 !important;margin:0.5rem 0;">Your Photo is Alive</h4>
        <p style="margin:0;">Time to make people say "whoa"</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="preview-container">', unsafe_allow_html=True)
    st.video(str(st.session_state.video_path))
    st.markdown('</div>', unsafe_allow_html=True)
    
    video_mb = st.session_state.video_path.stat().st_size / 1024 / 1024
    st.markdown(f'<p style="text-align:center;font-family:JetBrains Mono;color:#64748b;margin-top:0.5rem;">{video_mb:.1f} MB</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    
    with d1:
        with open(st.session_state.video_path, "rb") as f:
            st.download_button("⬇ Download Video", f.read(), "parallax_loop.mp4", "video/mp4", use_container_width=True)
    
    with d2:
        if st.session_state.gaussian_path:
            with open(st.session_state.gaussian_path, "rb") as f:
                st.download_button("⬇ Download 3D Model", f.read(), "model.ply", "application/octet-stream", use_container_width=True)
    
    with d3:
        if st.button("↺ Start Over", use_container_width=True):
            if st.session_state.work_dir:
                shutil.rmtree(st.session_state.work_dir, ignore_errors=True)
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div class="studio-footer">
    <span class="footer-logo">◈ Parallax Studio v{CONFIG['version']}</span>
    <br>
    <span>Your Photos. Alive.</span>
    <br>
    <span style="font-size:0.75rem;opacity:0.6;">Powered by Apple SHARP + Qwen-Image-Edit</span>
</div>
""", unsafe_allow_html=True)
