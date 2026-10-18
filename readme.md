# SGD Portable

![Explanation](./img/analyzed.jpg)

🎧 A portable, self-contained web interface for music genre classification using MAEST.

Two variants available:
- **CUDA version**: GPU acceleration via CUDA (NVIDIA)  
- **CPU version**: Runs on any Windows machine without GPU support  

No Python or system-wide installation required — everything runs from this folder.

---

## 📦 Installation Size

| Variant | Installed (after `start.bat`) |
|---------|---------------------------------|
| CPU     |    **~2 GB**                    |
| CUDA    |    **~6 GB**                    |

> ⚠️ MAEST model (~350 MB) downloads on first use — not included in the ZIP.

---

## ✅ Features

| Feature | Details |
|---------|---------|
| 📦 Portable | All-in-one folder: embedded Python 3.12.9, all dependencies pre-installed |
| 🖥️ No setup | Double-click `start.bat` → wait → launch. No admin rights needed. |
| 🚀 GPU support (CUDA) | Uses NVIDIA CUDA cu12.4 automatically when available; falls back to CPU otherwise |
| 🧠 Model | [MAEST](https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e-519l): genre & subgenre classification (519 styles, 30-second clips) |
| 🌐 Web UI | Gradio-based — dark/light theme, file upload, charts, tag cloud |

---

## ⚠️ License Notice

### Code
MIT License — see `LICENSE`.  
You may use, modify, and share this tool — even commercially.

### MAEST Model (CC BY-NC 4.0)
❌ Non-commercial only — attribution required  
✅ Allowed: research, education, personal use  
❌ Not allowed: commercial use of the model or its outputs  

The **tag cloud is SGD Portable’s own work**, not derived from MAEST — free to use.

Model is *not included* — downloads automatically on first use.  
See license: https://creativecommons.org/licenses/by-nc/4.0/

### Dependencies
All Python packages (`torch`, `gradio`, `transformers`, etc.) are open-source (BSD/Apache 2.0/MIT).  
Full details in `THIRD_PARTY_LICENSES.md`.

---

## 📥 Installation

1. Download the appropriate ZIP:
   - **CUDA version**: `SGD_Portable_CUDA.zip` *(NVIDIA GPU with CUDA required)*
   - **CPU version**: `SGD_Portable_CPU.zip` *(works anywhere, slower)*
> 💡 Note that the CUDA version will not install on a pc without fitting cuda support. Please use the CPU version instead.

2. Extract the folder.

3. Double-click `start.bat`.  
   → Downloads Python, installs ~1.5–6 GB of packages (depending on variant).  
   → Optionally downloads MAEST model (~350 MB).

4. Wait for confirmation → web UI starts automatically in your browser.

> 💡 Leave the terminal window open while using the app.

---

## 🎯 Usage
- Drag & drop audio files (MP3, WAV, etc.) into the upload zone  
- Click **▶ Analyze** to detect genre and subgenre  
- Toggle **Dark theme** / **Tag cloud**  
- View charts: main genres vs. detailed substyles
- To stop the software close the terminal window

> ⚠️ You have to clear the result to be able to drop the next audio files. But you can append to the file list by using the file icon and choose other files.

> ⚠️ Note that not all audio files might load, even when your favourite player plays them. When in trouble, try to resave it with an audio editor like Audacity.

> ⚠️ Tags are hardcoded. You can manually edit them if you want. The content is in the tagcloud,py

> ⚠️ Results are probabilistic — AI makes mistakes. Always verify.

## 📂 Batch
Analyse a whole folder and build a catalogue you can open in Excel.

- Under the charts, enter a folder or click **📁 Browse**
- Tick **Include subfolders** to go through every folder below it
- Click **▶ Start**. Tracks run one after another and the progress log shows each result
- Click **■ Stop** to halt after the current track

Supported files: WAV, MP3, FLAC, OGG, M4A, AIFF.

Results go into `sgd_catalogue.csv` in the chosen folder. Each track gets a row as soon as it is done:

- **file, folder** and **idea**: the name without take numbers and tags, so `ACID CELLULOID [E] 001` and `004` share the idea `ACID CELLULOID`
- **lane** and **artist**: which of your lanes the track fits, from `lanes.csv` (see below)
- top 3 **genres** and top 5 **subgenres** with %. Each style is scored on its own, so a genre's % is its strongest style and the numbers do not add up to 100
- **bpm**, **key**, **camelot** (the key on the Camelot wheel used for harmonic mixing), **length**, **loudness_db** and **energy** (1-10)
- **ending**: clean, fade, cut off or long silence. A rough check to spot tracks that stop dead
- tags, status, error, model and time of analysis

Every track's full set of style scores also goes into `sgd_fingerprints.jsonl` in the same folder, for grouping and album building later.

### Lanes
`lanes.csv` next to `genre_webui.py` sets out up to five lanes, one per artist name. Each row lists styles separated by `;` and an optional BPM range. A track goes to the lane with the strongest matching style, or `Unsorted`. Put your artist names in the `artist_name` column. Edits apply from the next track.

> 💡 A catalogue made by an older SGD version is renamed to `sgd_catalogue_old_<date>.csv` and the folder is analysed again from scratch.

> 💡 Press Start again on the same folder to carry on. Tracks already in the CSV are skipped. Tick **Retry tracks that failed before** to have another go at the failed ones.

> 💡 You can open the CSV in Excel while a batch runs. Excel locks the file, so SGD holds new rows in memory and saves them once you close it.

## 🎯 Uninstall
- Simply delete the folder. It is a virtual environment (Venv), and contains everything needed in its own folder.

---

## 🔧 System Requirements

| Requirement       | CUDA version            | CPU version            |
|-------------------|-------------------------|------------------------|
| OS                | Windows 10/11 (64-bit)  | Windows 10/11 (64-bit) |
| RAM               | ≥ 6 GB                  | ≥ 4 GB                 |
| GPU (optional)    | NVIDIA GPU + CUDA 12.x  | —                      |
| Storage           | ~6 GB free              | ~2 GB free             |

---

## 📜 Credits
- MAEST model: [MTG-UPF](https://mtg.upf.edu/) (Universitat Pompeu Fabra)  
- WebUI & tooling: SGD Portable  

Made with ❤️ by Reiner Prokein 
https://haizytiles.reinerstilesets.de/

---

> 📬 Feedback? Bugs? Contributions? Open an issue or PR!
