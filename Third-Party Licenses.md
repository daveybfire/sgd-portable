# Third-Party Licenses

This software distribution includes or depends on the following open-source components.  
All dependencies are installed at runtime via `pip install --target=...`, except where noted.

| Component                | License                    | Copyright / Notes                                                                 |
|--------------------------|----------------------------|-----------------------------------------------------------------------------------|
| **Your WebUI code**      | [MIT](LICENSE)             | © [Jahr] [Dein Name/Alias]                                                       |
| `torch` / `torchaudio`   | [BSD 3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) | © 2016–2025 Facebook / PyTorch Contributors                                      |
| `transformers`           | [Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) | © 2018–2025 HuggingFace Inc. — requires attribution                             |
| `librosa`                | [ISC](https://github.com/librosa/librosa/blob/main/LICENSE.md) | © 2013–2025 librosa contributors                                                |
| `soundfile`              | [BSD 3-Clause](https://github.com/bastibe/python-soundfile/blob/master/LICENSE) | © 2014–2025 Bastian Bechtold                                                   |
| `gradio`                 | [Apache 2.0](https://github.com/gradio-app/gradio/blob/main/LICENSE.txt) | © 2023 Gradio Inc.                                                               |
| `plotly.py`              | [MIT](https://github.com/plotly/Plotly.Python/blob/master/LICENSE.txt) | © 2025 Plotly, Inc.                                                              |
| `numpy`, `pandas`        | [BSD 3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) / [BSD 3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE) | © NumPy / pandas Project contributors                                           |
| **MAEST model** *(see below)* | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | © 2023 Music Technology Group, Universitat Pompeu Fabra — *non-commercial only* |
| **Audiobox Aesthetics** (`facebook/audiobox-aesthetics`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | © Meta Platforms, Inc. Quality scores. Downloaded on first use |
| **CLAP model** (`laion/larger_clap_music`) | [Apache 2.0](https://huggingface.co/laion/larger_clap_music) | © LAION. Vocal, mood and instrument tags and text search. Downloaded on first use |
| `pyloudnorm`             | [MIT](https://github.com/csteinmetz1/pyloudnorm/blob/master/LICENSE) | © Christian Steinmetz. LUFS loudness |
| `fpcalc` (Chromaprint)   | [LGPL 2.1](https://github.com/acoustid/chromaprint/blob/master/LICENSE.md) | © Lukáš Lalinský. Downloaded by the start script into `bin` |
| AcoustID web service     | [Terms](https://acoustid.org/webservice) | Optional. Free for non-commercial use; commercial use needs a paid plan from acoustid.biz |

> ⚠️ **Important**: The MAEST model (`mtg-upf/discogs-maest-10s-fs-129e`) is **not included** in this package.  
> It is downloaded automatically on first use from Hugging Face.  
> Its license (**CC BY-NC 4.0**) prohibits *commercial use*. Attribution is required.

### MAEST Model License Notice
You may:
- ✅ Share — copy and redistribute the material in any medium or format  
- ✅ Remix — transform and build upon the material for **non-commercial purposes only**  

Under the following terms:
- 🔔 **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

You may not use the material for commercial purposes.
See full license: https://creativecommons.org/licenses/by-nc/4.0/
