# SignVision: Real-Time Sign Language Conversion

SignVision watches **Indian Sign Language (ISL)** through an ordinary webcam and writes it out as text in real
time, with optional speech output. It recognises

- **letters and numbers** (A–Z, 1–9) held still, one frame at a time, and
- **words** (hello, thank you, …) from about 2 seconds of hand movement, and
- **whole sentences** (101 everyday sentences such as "Hi, how are you?", "Where are you from?") signed in
  one movement, for a two-way **Conversation**: Person A signs, Person B answers by voice (offline speech-to-text).

Everything runs locally on a normal laptop (no GPU, no paid APIs) using only open-source tools.

Main project, Batch B12, Department of IT, Seshadri Rao Gudlavalleru Engineering College (2026–27).
Guide: Ms. Sykam Vani.

---

## 1. Install (Windows, one time)

You need **Miniconda** and an internet connection. Nothing else: Python, Node.js and every library are
installed by the setup script.

1. Install Miniconda: <https://docs.conda.io/en/latest/miniconda.html> (Windows 64-bit installer, default options).
2. Download or clone this project folder, e.g. to `D:\signvision`.
3. Open **Anaconda Prompt** from the Start menu and run:

   ```bat
   cd /d D:\signvision
   setup.bat
   ```

   The first run downloads about 1.5 GB and takes 5–15 minutes. You can also just double-click `setup.bat`.

## 2. Run

Double-click **`run.bat`** (or run it in Anaconda Prompt). Your browser opens **<http://localhost:8000>**.
Keep the black window open while you use the app; close it to stop.

Use **Chrome or Edge**, and allow camera access when the browser asks.

> Linux / macOS / WSL: `./setup.sh` once, then `./run.sh`.

## 3. Using the app

| Page | What it does |
|---|---|
| **Translate** | Live webcam translation. Choose *Letters* or *Words*. In letters mode, hold a sign until the ring fills; lower your hands to add a space. Buttons (or Space / Backspace keys) edit the text, *Speak* reads it aloud, *Save* stores it. |
| **Conversation** | Two people, one laptop. Person A signs a whole sentence (raise hands, sign, lower hands); it appears in the chat and is spoken aloud. Person B taps the mic and speaks; offline Whisper writes it into the chat. *Did you mean…* buttons appear when the app is unsure. |
| **Invite Person B (rooms)** | On Conversation, press **Invite Person B**. The laptop creates a room and a free, temporary internet link (Cloudflare Quick Tunnel, no account) with a QR code. Person B opens it on their own phone from anywhere: Person A's signed sentences appear there and are read aloud, and Person B replies by speaking or typing. The link only opens that chat; the rest of the app stays private. It ends when SignVision is closed. |
| **Sign guide** | Video of every sentence and word sign (from the datasets) and photos of every letter, so users can learn them. |
| **Photo or video** | Upload a photo (read as one letter) or a video (read as letters or words, with a timeline). |
| **Record and train** | Record your own examples of any sign and train the models with one click. |
| **History** | Saved translations: copy, speak or delete. |
| **About** | How the system works, model accuracy, comparison tables and all result figures. |

Both models come **pre-trained**, so the app works straight after setup:

| Model | Trained on | Test accuracy |
|---|---|---|
| Letters and numbers (35 signs) | 42,667 public ISL images | 99.4 % |
| Words (9 greetings: hello, how are you, alright, good morning/afternoon/evening/night, thank you, pleased) | 190 INCLUDE videos from several signers | 95.4 % ± 3.8 (mean of 6 runs) |
| Sentences (101 ISL-CSLTR sentences + a "not a sign" class) | 682 videos, ~5 for training per sentence | 46.0 % incl. the "not a sign" class (61.9 % within top 3; best baseline 29.4 %). With the live rules (≥ 60 % sure and 25 points ahead): 25 % of unseen sentences typed correctly, 11 % typed wrong, 64 % left as "Did you mean…?"; random movements typed as a sentence 4 % (34 % before the "not a sign" class). |

The sentence model is limited by data: the public corpus has only about 7 videos per sentence. Recording each
sentence 5–10 times on **Record and train → Sentences** (ideally by several team members) and retraining makes it
much more reliable for your own signing. For a demo, pick the sentences you need and record those.

The public data is already included in the repo as hand landmarks (`backend/data/processed/`, 16 MB), so
retraining always combines it with your own recordings.

## 4. Training on your own signs (recommended)

Models trained on one dataset often struggle with new people, cameras and rooms. Recording yourselves fixes that
and gives you original data for the paper.

1. Open **Record and train**, press *Start camera*, type your name under *Signer*.
2. Pick a sign, press *Record*. After a 3-2-1 countdown:
   - **Letters**: hold the sign for 3 s, moving it slightly.
   - **Words**: perform the whole sign once within 2.5 s.
3. Repeat. Aim for **5+ recordings per letter** and **15+ per word**, spread across all four team members.
   Also record **"No sign (idle)"** in Words mode (hands resting or moving naturally): it stops the app from typing
   random words when nobody is signing.
4. Press **Train letters model** / **Train words model**. Training takes 1–3 minutes; the new model is used
   immediately and its results appear on the About page. New words (e.g. *doctor*, *water*) are added to the
   words model as soon as they have recordings.

Only hand-landmark coordinates are saved (`backend/data/own/`), never video.

### Adding more public data

- **Letters:** `get_public_data.bat` re-downloads the original ISL images (~290 MB) and converts them again.
  Only needed if you change the feature extraction.
- **More words (INCLUDE dataset, AI4Bharat, CC-BY-4.0):** download more category zips from
  <https://zenodo.org/records/4010759> (e.g. `Pronouns_1of2.zip`, `Days_and_Time_1of3.zip`), unzip them into one
  folder, then in Anaconda Prompt:

  ```bat
  conda activate signvision
  cd backend
  python -m scripts.prepare_videos --src "D:\datasets\INCLUDE" --name include
  python -m scripts.train_dynamic
  ```

## 5. Command-line reference

Run these inside `backend/` after `conda activate signvision`.

| Command | Purpose |
|---|---|
| `python -m scripts.download_isl_images` | Download the public ISL alphabet images |
| `python -m scripts.prepare_images --src <folder> --name <name>` | Images (one sub-folder per sign) → landmarks |
| `python -m scripts.prepare_videos --src <folder> --name <name>` | Videos (one folder per word) → landmark clips |
| `python -m scripts.train_static` | Train the letters/numbers model (+ SVM, RF, k-NN, LogReg baselines) |
| `python -m scripts.train_dynamic` | Train the words model (GRU and LSTM, best one kept, + baselines) |
| `python -m scripts.train_dynamic --kind sentence` | Train the sentences model |
| `python -m scripts.prepare_videos --kind sentence --src <folder> --name <name>` | Sentence videos (one folder per sentence) → landmark clips |
| `python -m scripts.make_sign_clips` | Rebuild the Sign guide videos from the datasets |
| `python -m scripts.download_speech_model` | Download the offline speech-to-text model (done by setup) |
| `python -m scripts.evaluate_cross --kind static` | Cross-dataset test: train on public data, test on your recordings |

Add `--help` to any command for options (e.g. `--epochs`, `--no-public`, `--no-own`).
`train.bat` trains both models in one go. `dev.bat` starts hot-reloading servers for frontend development.

Every training run writes figures and numbers for the paper to `backend/reports/<static|dynamic>/`:
`confusion_matrix.png`, `training_curves.png`, `model_comparison.png`, `per_class_f1.png`,
`dataset_distribution.png`, `classification_report.txt` and `metrics.json`.

## 6. How it works

```
Browser (React)                         Backend (FastAPI, Python)
┌────────────────┐   JPEG frames    ┌──────────────────────────────────────────────┐
│ webcam capture │ ───────────────▶ │ OpenCV pre-processing (resize, low-light     │
│ skeleton view  │    WebSocket     │   CLAHE)                                     │
│ transcript     │ ◀─────────────── │ MediaPipe HandLandmarker → 2 × 21 3-D points │
│ speech output  │  sign + text     │ features: wrist-centred, scale-normalised    │
└────────────────┘                  │ Letters: MLP on one frame                    │
                                    │ Words:   Bi-GRU/LSTM + attention on 30 frames│
                                    │ smoothing + sentence builder, SQLite history │
                                    └──────────────────────────────────────────────┘
```

- **Features**: each hand is shifted so the wrist is at the origin and scaled to unit size, making recognition
  independent of distance and position; the hand-to-hand vector captures two-handed ISL signs. Words also use
  wrist positions over time.
- **Augmentation** during training: mirroring (left-handed signers), rotation, scaling, shifting, landmark noise,
  and for words time-warping and dropped frames.
- **Honest evaluation**: train/test splits keep whole recordings (and blocks of consecutive dataset images)
  on one side, so near-duplicate frames never leak into the test set.

## 7. Project structure

```
signvision/
├── setup.bat / run.bat / dev.bat / train.bat / get_public_data.bat   Windows one-click scripts
├── setup.sh / run.sh                                                 Linux/macOS equivalents
├── environment.yml                     Python 3.11 + Node.js 20 + all pinned libraries
├── backend/
│   ├── app/            FastAPI server: REST API, WebSockets, history DB, background training
│   ├── signvision/     ML package: hands.py, features.py, models.py, engine.py, data.py, trainer.py
│   ├── scripts/        dataset download/preparation and training scripts
│   ├── assets/         MediaPipe hand_landmarker.task model
│   ├── models/         trained models (static_model.pt, dynamic_model.pt)
│   ├── reports/        metrics and figures from training
│   └── data/           recordings, processed datasets, SQLite database (not in git)
├── frontend/           React + TypeScript + Tailwind web app (Vite)
└── docs/               IEEE paper and project report
```

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `conda` is not recognized | Use **Anaconda Prompt** from the Start menu, not normal Command Prompt. |
| Camera doesn't start | Allow camera in the browser (camera icon in the address bar); close Zoom/Teams/Camera apps; use `http://localhost:8000`, not an IP address. |
| "Can't reach the SignVision server" | `run.bat` window must stay open. If port 8000 is busy, close the other program or restart the laptop. |
| Random words typed when nobody signs | Record "No sign (idle)" 15+ times in Words mode, then train the words model. |
| Letters are typed wrongly | Good light, plain background, hands fully in view; record your own samples and retrain. |
| Slow (< 8 fps) | Close other heavy apps; plug in the charger (power-saving mode slows the CPU). |
| `DLL load failed` when the server starts | Install the Microsoft Visual C++ Redistributable (x64): <https://aka.ms/vs/17/release/vc_redist.x64.exe>, then run `run.bat` again. |
| `setup.bat` fails halfway | Check internet and run it again; it continues where it stopped. |

## 9. Technology (all open source)

Frontend: React 19, TypeScript, Vite, Tailwind CSS, lucide icons, Atkinson Hyperlegible Next font, Web Speech API.
Backend: Python 3.11, FastAPI, Uvicorn, WebSockets, SQLAlchemy + SQLite.
ML: MediaPipe Hand Landmarker (Apache-2.0), OpenCV, PyTorch, scikit-learn, NumPy, pandas, Matplotlib, seaborn.
Speech-to-text: faster-whisper (MIT) with the Whisper base.en model (MIT), running offline on the CPU.

## 10. Data sources and credits

- ISL alphabet images (35 classes): Kaggle "Indian Sign Language (ISL)" dataset, via the Hugging Face mirror
  `Hemg/Indian_sign_language_dataset`. Cite the original authors in the paper.
- INCLUDE: A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra, "INCLUDE: A Large Scale Dataset for Indian Sign
  Language Recognition", ACM Multimedia 2020. CC-BY-4.0.
- ISL-CSLTR: R. Elakkiya, B. Natarajan, "ISL-CSLTR: Indian Sign Language Dataset for Continuous Sign Language
  Translation and Recognition", Mendeley Data, V1, 2021, doi:10.17632/kcmpdxky7p.1. CC BY 4.0. Used for the
  sentence model and the Sign guide videos (shortened and resized).
- MediaPipe Hand Landmarker model, Google, Apache-2.0.
- Whisper (OpenAI, MIT) via faster-whisper (SYSTRAN, MIT).
- cloudflared (Cloudflare, Apache-2.0) for the room links; qrcode-generator (Kazuhiko Arase, MIT) for the QR code.
