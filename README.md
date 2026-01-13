# GEVWallMagicV2
Transform photos into depth-enhanced video wall content
# ◈ Parallax Studio v1.2

### *Your Photos. Alive.*

---

## What Is This?

You give it a photo. Any photo.

Seconds later, it's no longer flat. It *breathes*. Objects in the front drift differently than objects in the back. Your brain stops seeing a screen and starts seeing a *place*.

People will stop. They'll tilt their heads. They'll say **"wait... is that 3D?"**

It's not. But it *feels* like it is.

---

## The Magic (Optional)

Before we add depth, you can transform your photo into something else entirely.

**One click:** your office courtyard becomes a **Studio Ghibli dreamscape**

**One click:** that ordinary skyline becomes a **moody oil painting**

**One click:** your campus quad glows with **golden hour light** it never actually had

*Then* we add the depth. Now you have a living painting. Art that moves. A window into a world that never existed — but looks like it should.

---

## Quick Start

### Step 1: Install (One Time)

Double-click `install.bat` and wait 20-40 minutes.

That's it. Everything installs automatically.

### Step 2: Run

Double-click **"Parallax Studio"** on your Desktop.

Or double-click `run.bat` in the app folder.

### Step 3: Create

1. Upload a photo
2. (Optional) Pick an art style
3. Choose your screen size
4. Click a few buttons
5. Download a video that makes people say "whoa"

---

## What's In The Box

```
ParallaxStudio/
├── parallax_studio.py    ← The app
├── install.bat           ← Run this first
├── run.bat               ← Run this to start
└── README.md             ← You're reading it
```

**⚠️ Keep all files in the same folder. Moving them individually breaks things.**

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 3090 (24GB) | RTX 4090/5090 |
| **RAM** | 32GB | 64GB+ |
| **Storage** | 60GB free | 100GB+ SSD |
| **OS** | Windows 10 | Windows 11 |

**Why so beefy?**

The style transformation (Qwen) uses a 20-billion parameter AI model. The 3D extraction (SHARP) needs fast GPU memory. Your RTX 5090 will handle it beautifully.

---

## The Technology (For the Curious)

### Apple SHARP
Cutting-edge research that looks at a single photo and understands its 3D structure. In under a second, it builds a complete 3D model using something called "Gaussian Splatting."

### Qwen-Image-Edit
A 20-billion parameter AI from Alibaba that can transform your photo into any artistic style — Ghibli, oil painting, cyberpunk — while preserving its structure.

### The Parallax Effect
When you move your head in real life, nearby things shift more than distant things. The video simulates a camera gently drifting. Because we know the 3D structure, objects move at different speeds. Your brain sees depth.

---

## Style Presets

| Style | What It Does |
|-------|--------------|
| **Studio Ghibli** | Soft anime aesthetic with whimsical atmosphere |
| **Oil Painting** | Classical brushstrokes and rich colors |
| **Watercolor** | Soft edges, translucent colors |
| **Cyberpunk** | Neon lights, futuristic vibes |
| **Golden Hour** | Warm sunset lighting with long shadows |
| **Dramatic Sky** | Replaces boring skies with epic sunsets |
| **Impressionist** | Monet-style visible brushwork |
| **Noir** | Black and white with dramatic shadows |
| **Add Fog** | Atmospheric mist for mystery and depth |

Or write your own custom prompt.

---

## Tips for Best Results

### ✓ DO

- Use photos with **clear depth layers** (foreground, middle, background)
- **Landscapes** and **cityscapes** work amazingly
- **Higher resolution** input = better output
- Start with **subtle depth settings** (0.10-0.15)
- Use **"Dramatic Sky"** or **"Add Fog"** to enhance depth perception

### ✗ DON'T

- Flat subjects (documents, walls, whiteboards)
- Extreme close-ups of faces
- Images with lots of transparency or reflections

---

## Display Formats

| Aspect Ratio | Use Case |
|--------------|----------|
| **16:9** | Standard TVs, monitors, presentations |
| **21:9** | Ultrawide monitors, cinematic displays |
| **32:9** | Video walls, Samsung Odyssey, dual-screen setups |
| **Custom** | Whatever weird size you need |

---

## Troubleshooting

### "NVIDIA GPU not detected"
- Install latest drivers from [nvidia.com/drivers](https://nvidia.com/drivers)
- Restart computer after installing
- Make sure monitor is plugged into GPU, not motherboard

### "Conda not recognized"
- Restart computer after installing Miniconda
- Run `install.bat` again

### "Out of memory"
- Close other GPU applications
- Use "Half" or "Preview" resolution
- Skip style enhancement (it uses a lot of VRAM)

### "Qwen download is slow/stuck"
- It's a 40GB model — be patient on first run
- Check your internet connection
- Try again later if servers are busy

### App won't open in browser
- Manually go to: http://localhost:8501
- Check if port 8501 is in use by another app

---

## The Bottom Line

You're not running software.

You're buying reactions.

The "whoa."
The double-take.
The pull-out-your-phone-and-record moment.

One photo. A few clicks. A display people remember.

---

## Credits

- **Apple SHARP** — [arxiv.org/abs/2512.10685](https://arxiv.org/abs/2512.10685)
- **Qwen-Image-Edit** — [Alibaba Qwen Team](https://huggingface.co/Qwen/Qwen-Image-Edit)
- **Built with** — Streamlit, PyTorch, gsplat, FFmpeg

---

<p align="center">
<b>◈ Parallax Studio v1.2</b><br>
<i>Your Photos. Alive.</i>
</p>
