# 🐦 Birdsong Classification with Python

**An introduction to computational notebooks and data science — for animal behavior students.**

This project teaches you how to process, visualize, and analyze bird vocalizations using Python. You'll go from zero programming experience to training a neural network that clusters bird songs by species — all inside Google Colab, with nothing to install on your own computer.

The dataset includes 98 field recordings across five species: Bachman's sparrow, Northern cardinal, Song sparrow, Swamp sparrow, and Zebra finch.

---

## 🪺 Why Birdsong?

Bird vocalizations are one of the richest and most accessible windows into animal behavior. Songs encode information about species identity, individual recognition, mate quality, territory boundaries, and environmental conditions. Historically, analyzing these signals required trained ears and painstaking manual transcription.

Today, the same questions can be explored computationally — and this project introduces you to the tools that make that possible:

- **Audio signal processing** — loading, visualizing, and manipulating sound as data
- **Feature extraction** — transforming raw audio into meaningful numerical representations (MFCCs)
- **Unsupervised learning** — using autoencoders to discover structure in data without labeled examples
- **Scientific visualization** — turning numbers into spectrograms, waveforms, and scatter plots that tell a story

These are foundational skills in bioacoustics, but they transfer directly to any data-intensive research.

---

## 🗂️ Notebook Sequence

The project is organized as a series of four notebooks, designed to be worked through in order. Each one builds on the last.

### Notebook 0 — A Naturalist's Field Notebook in Python
> *No programming experience required.*

Learn Python fundamentals through the lens of field biology. Variables are specimen labels, lists are species checklists, loops walk a trail, and charts sketch what you observe. By the end you'll understand the building blocks used in every notebook that follows.

**Concepts:** variables, strings, lists, dictionaries, loops, functions, conditionals, plotting

### Notebook 1 — Birdsong Introduction
> *Your first encounter with real audio data.*

Clone the dataset from GitHub, explore one recording from each species (waveform, spectrogram, playback), then load all 98 files and visualize them as spectrogram montages. Finally, extract MFCC features and train an autoencoder to produce a 2D scatter plot showing how species cluster.

**Concepts:** librosa, spectrograms, MFCCs, tensors, autoencoders, embedding visualization

### Notebook 2 — Downloading & Exploring Audio from the Web
> *Working with data from online sources.*

Download birdsong recordings from public URLs, visualize and compare them side by side, and learn how to fetch audio from any web source. Includes a section where you paste your own URL to explore.

**Concepts:** HTTP requests, file I/O, comparative visualization

### Notebook 3 — Analyze Your Own Data
> *Bring your own recordings.*

Upload audio files you've collected (phone recordings, field equipment, any format) to Google Drive, then process, visualize, and cluster them with the same pipeline from Notebook 1. This is where the tools become yours.

**Concepts:** Google Drive integration, pydub, multi-format audio, self-directed analysis

---

## 🚀 Getting Started

1. Open any notebook in **Google Colab** (no local installation needed):
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - **File → Open notebook → GitHub**
   - Paste this repository URL and select a notebook

2. **Save your own copy:** File → Save a copy in Drive

3. **Run cells in order** from top to bottom — click the Play button or press **Shift + Enter**

Notebooks 0–2 are fully self-contained and require nothing beyond a web browser. Notebook 3 uses Google Drive for your own audio uploads.

---

## 📁 Repository Structure

```
├── Notebook_0_Python_Basics.ipynb      # Python fundamentals
├── Notebook_1_Birdsong_Intro.ipynb     # Audio processing & ML pipeline
├── Notebook_2_Web_Audio.ipynb          # Downloading & exploring web audio
├── Notebook_3_Your_Own_Data.ipynb      # Bring your own recordings
├── Bachman_s sparrow/                  # 19 WAV files
├── Northern cardinal/                  # 9 WAV files
├── Song sparrow/                       # 15 WAV files
├── swamp sparrow/                      # 10 WAV files
├── Zebra finch/                        # 45 WAV files
├── archive/                            # Previous notebook versions
└── README.md
```

---

## 🔬 What You'll Learn

By the end of this project, you will be able to:

- Write and run Python code in a computational notebook
- Load audio files and understand sample rates, amplitude, and duration
- Read waveforms and spectrograms — two fundamental representations of sound
- Extract MFCC features that capture the timbral character of a vocalization
- Train an autoencoder neural network to compress high-dimensional data into 2D
- Interpret embedding plots to identify clustering patterns across species
- Process your own audio data end-to-end

---

## 🎓 For Instructors

This project is designed as a multi-session lab module for undergraduate or early graduate students in animal behavior, ecology, or related fields. No prior programming experience is assumed.

**Suggested pacing:**
- **Session 1:** Notebook 0 (Python basics) — 45–60 min
- **Session 2:** Notebook 1 (birdsong pipeline) — 60–90 min
- **Session 3:** Notebook 2 (web audio) + Notebook 3 (own data) — 60–90 min

The notebooks are self-paced and include explanations at every step. Students are encouraged to modify code, re-run cells, and experiment — that's how computational thinking develops.

---

## 🐤 Species in the Dataset

| Species | Files | Notes |
|---------|------:|-------|
| Bachman's sparrow (*Peucaea aestivalis*) | 19 | Field recordings with coded identifiers |
| Northern cardinal (*Cardinalis cardinalis*) | 9 | Named individuals with distinct song types |
| Song sparrow (*Melospiza melodia*) | 15 | Multiple song types per population |
| Swamp sparrow (*Melospiza georgiana*) | 10 | Recordings from Conneaut and Valley sites |
| Zebra finch (*Taeniopygia guttata*) | 45 | Lab recordings including train/test splits by individual |

---

## 📦 Dependencies

All dependencies are pre-installed in Google Colab or installed automatically by the notebooks:

- Python 3
- NumPy, Matplotlib
- librosa (audio processing)
- PyTorch (neural networks)
- scikit-learn (preprocessing)
- pydub (multi-format audio, Notebook 3 only)

---

## 📝 License

Audio data and notebook content are provided for educational use.
