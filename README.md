# Birdsong Classification with Python

**An introduction to computational notebooks and data science — for animal behavior students.**

This project teaches you how to process, visualize, and analyze bird vocalizations using Python. You'll go from zero programming experience to training a neural network that clusters bird songs by species — all inside Google Colab, with nothing to install on your own computer.

The dataset includes 98 field recordings across five species: Bachman's sparrow, Northern cardinal, Song sparrow, Swamp sparrow, and Zebra finch.

---

## Why Birdsong?

Bird vocalizations are one of the richest and most accessible windows into animal behavior. Songs encode information about species identity, individual recognition, mate quality, territory boundaries, and environmental conditions. Historically, analyzing these signals required trained ears and painstaking manual transcription.

Today, the same questions can be explored computationally — and this project introduces you to the tools that make that possible:

- **Audio signal processing** — loading, visualizing, and manipulating sound as data
- **Feature extraction** — transforming raw audio into meaningful numerical representations (MFCCs)
- **Unsupervised learning** — using autoencoders to discover structure in data without labeled examples
- **Scientific visualization** — turning numbers into spectrograms, waveforms, and scatter plots that tell a story

These are foundational skills in bioacoustics, but they transfer directly to any data-intensive research.

---

## Getting Started

Each notebook has an **"Open in Colab"** button — click it to open the notebook in Google Colab, where you can run the code directly in your browser with nothing to install. **You must use these buttons to run the notebooks.** Do not try to download them or open them on your computer — they are designed to run in Colab.

Once a notebook opens in Colab, go to **File → Save a copy in Drive** so you have your own version to work in.

---

## Notebook Sequence

The project is organized as a series of six notebooks, designed to be worked through in order. Each one builds on the last.

### Notebook 0 — A Naturalist's Field Notebook in Python [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/williamedwardhahn/birdsong/blob/main/Notebook_0_Python_Basics.ipynb)
> *No programming experience required.*

Learn Python fundamentals through the lens of field biology. Variables are specimen labels, lists are species checklists, loops walk a trail, and charts sketch what you observe. Ends with a teaser — loading and hearing a real birdsong recording.

**Concepts:** variables, strings, lists, dictionaries, loops, functions, conditionals, plotting

### Notebook 1 — Sound as Data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/williamedwardhahn/birdsong/blob/main/Notebook_1_Sound_as_Data.ipynb)
> *The bridge from Python to audio.*

Take one bird, one recording, and build understanding from the ground up. What is digital audio? What does sample rate mean? What is a spectrogram and why does it matter? By the end you'll be able to look at a spectrogram and *read* what a bird is doing.

**Concepts:** digital audio, sample rate, waveforms, spectrograms, audio manipulation

### Notebook 2 — Birdsong Species Explorer [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/williamedwardhahn/birdsong/blob/main/Notebook_2_Birdsong_Explorer.ipynb)
> *Meet all five species — with your eyes and ears.*

Listen to recordings from each species, compare their spectrograms side by side, view montages of all 98 files, and test whether *you* can identify species from unlabeled spectrograms. No machine learning — just pattern recognition.

**Concepts:** comparative listening, spectrogram reading, within-species variation, visual pattern recognition

### Notebook 3 — Clustering Birdsong with Machine Learning [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/williamedwardhahn/birdsong/blob/main/Notebook_3_Clustering.ipynb)
> *Teach a computer to hear the differences.*

Extract MFCC features that capture the texture of each recording, train an autoencoder to compress them into 2D, and visualize how species cluster. Then experiment: change the bottleneck size, learning rate, and number of epochs to see how each affects the results.

**Concepts:** MFCCs, autoencoders, bottleneck compression, learning rate, training epochs, clustering

### Notebook 4 — Downloading & Exploring Audio from the Web [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/williamedwardhahn/birdsong/blob/main/Notebook_4_Web_Audio.ipynb)
> *Working with data from online sources.*

Download birdsong recordings from public URLs, visualize and compare them side by side, and learn how to fetch audio from any web source. Includes a section where you paste your own URL to explore.

**Concepts:** HTTP requests, file I/O, comparative visualization

### Notebook 5 — Analyze Your Own Data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/williamedwardhahn/birdsong/blob/main/Notebook_5_Your_Own_Data.ipynb)
> *Bring your own recordings.*

Upload audio files you've collected (phone recordings, field equipment, any format) to Google Drive, then process, visualize, and cluster them with the same pipeline from Notebook 3. This is where the tools become yours.

**Concepts:** Google Drive integration, pydub, multi-format audio, self-directed analysis

---

## Getting Started

1. Open any notebook in **Google Colab** (no local installation needed):
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - **File → Open notebook → GitHub**
   - Paste this repository URL and select a notebook

2. **Save your own copy:** File → Save a copy in Drive

3. **Run cells in order** from top to bottom — click the Play button or press **Shift + Enter**

Notebooks 0–4 are fully self-contained and require nothing beyond a web browser. Notebook 5 uses Google Drive for your own audio uploads.

---

## Repository Structure

```
├── Notebook_0_Python_Basics.ipynb        # Python fundamentals
├── Notebook_1_Sound_as_Data.ipynb        # Audio basics — the bridge
├── Notebook_2_Birdsong_Explorer.ipynb    # Explore 5 species by ear and eye
├── Notebook_3_Clustering.ipynb           # ML pipeline with experiments
├── Notebook_4_Web_Audio.ipynb            # Download & explore web audio
├── Notebook_5_Your_Own_Data.ipynb        # Bring your own recordings
├── bachmans_sparrow/                     # 19 WAV files
├── northern_cardinal/                    # 9 WAV files
├── song_sparrow/                         # 15 WAV files
├── swamp_sparrow/                        # 10 WAV files
├── zebra_finch/                          # 45 WAV files
├── archive/                              # Previous notebook versions
└── README.md
```

---


## Species in the Dataset

| Species | Files | Notes |
|---------|------:|-------|
| Bachman's sparrow (*Peucaea aestivalis*) | 19 | Field recordings with coded identifiers |
| Northern cardinal (*Cardinalis cardinalis*) | 9 | Named individuals with distinct song types |
| Song sparrow (*Melospiza melodia*) | 15 | Multiple song types per population |
| Swamp sparrow (*Melospiza georgiana*) | 10 | Recordings from Conneaut and Valley sites |
| Zebra finch (*Taeniopygia guttata*) | 45 | Lab recordings including train/test splits by individual |

---

## Dependencies

All dependencies are pre-installed in Google Colab or installed automatically by the notebooks:

- Python 3
- NumPy, Matplotlib
- librosa (audio processing)
- PyTorch (neural networks)
- scikit-learn (preprocessing)
- pydub (multi-format audio, Notebook 5 only)

---

## Collaborators

This project is a collaboration between:

- **[Rindy Anderson](https://biology.fau.edu/directory/anderson/index.php)** — Associate Professor, Department of Biological Sciences, Florida Atlantic University
- **[William Hahn](https://www.math.fau.edu/people/faculty/william-hahn.php)** — Associate Professor, Department of Mathematics and Statistics, Florida Atlantic University

---

## License

Audio data and notebook content are provided for educational use.
