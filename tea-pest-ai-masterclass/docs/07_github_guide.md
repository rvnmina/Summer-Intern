# Phase 5 — Project Management: Structuring & Publishing on GitHub

This guides you from "a folder on my laptop" to "a clean, professional GitHub repository" your
professor can browse. It assumes **zero** prior Git experience.

---

## 1. What Git and GitHub are (30-second version)
- **Git** = a tool on your computer that records **snapshots** ("commits") of your project so you can
  track changes and never lose work.
- **GitHub** = a website that hosts your Git project online so others can see it and you have a backup.
- **Repository (repo)** = one project's folder tracked by Git.

## 2. Recommended repository structure

This is the structure of *this very deliverable* — copy it. A clean layout signals seriousness.

```
tea-pest-ai-masterclass/
├── README.md                 # front page: what this is + how to navigate (already written)
├── LICENSE                   # e.g. MIT — lets others use your code
├── .gitignore                # tells Git which files to NOT upload (venv, caches, big data)
├── requirements.txt          # exact Python packages to install
├── docs/                     # the written masterclass (Phases 1–5)
│   ├── 01_reading_order.md
│   ├── 02_paper1_soc_dpu.md
│   ├── 03_paper3_edge_multimodal.md
│   ├── 04_paper2_tea_physio_fl.md
│   ├── 05_synthesis.md
│   ├── 06_future_work_and_skills.md
│   └── 07_github_guide.md
├── code/                     # runnable reference implementations
│   ├── paper1_mobilenetv3_transfer_learning.py
│   ├── paper2_chlorophyll_multimodal_transformer.py
│   ├── paper3_cnn_transformer_fusion.py
│   └── shared_concepts/
│       ├── attention_from_scratch.py
│       ├── quantization_and_distillation.py
│       └── federated_learning_demo.py
├── notebooks/                # (optional) Jupyter notebooks for experiments
├── figures/                  # diagrams/plots for your presentation
└── slides/                   # (optional) your final .pptx / PDF deck
```

**Why this layout:** separates *writing* (`docs/`) from *code* (`code/`) from *artifacts*
(`figures/`, `slides/`). Anyone landing on the repo reads `README.md`, then dives where they want.

## 3. One-time setup (do this once ever)

1. **Install Git:** download from <https://git-scm.com/downloads> (Windows/Mac/Linux).
2. **Make a free GitHub account:** <https://github.com>.
3. **Tell Git who you are** (in a terminal):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "ravindramina25@kgpian.iitkgp.ac.in"
   ```

## 4. Create the essential support files

**`.gitignore`** (prevents uploading junk/huge files) — create it at the repo root:
```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/
# Data & models (usually too big for Git)
data/
*.h5
*.pt
*.onnx
*.xmodel
# OS / editor
.DS_Store
.vscode/
.idea/
```

**`requirements.txt`** (already in this repo's `code/`), listing packages:
```
torch>=2.0
torchvision>=0.15
numpy
scikit-learn
opencv-python
matplotlib
```

**`LICENSE`** — the easiest is MIT. On GitHub you can auto-add it (Step 6), or paste the MIT text from
<https://choosealicense.com/licenses/mit/>.

## 5. Turn your folder into a Git repo and make the first commit

Open a terminal, `cd` into the project folder, then:
```bash
cd tea-pest-ai-masterclass

git init                       # start tracking this folder with Git
git add .                      # stage ALL files for the first snapshot
git status                     # (optional) see what will be committed
git commit -m "Initial commit: masterclass docs + reference code"
```
> A **commit message** is a short note describing the change. Use the imperative mood ("Add fusion
> demo", "Fix attention scaling").

## 6. Create the repo on GitHub and push

**Option A — via the website (easiest for beginners):**
1. On GitHub, click **+ → New repository**.
2. Name it `tea-pest-ai-masterclass`, add a short description, keep it **Public** (or Private if you
   prefer), **do NOT** check "Add a README" (you already have one). Click **Create repository**.
3. GitHub shows commands under *"…or push an existing repository"*. Copy them; they look like:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/tea-pest-ai-masterclass.git
   git push -u origin main
   ```
   - `git remote add origin …` links your local repo to the GitHub one.
   - `git push` uploads your commits. GitHub may ask you to log in / use a **Personal Access Token**
     (Settings → Developer settings → Personal access tokens → generate one, use it as the password).

Refresh the GitHub page — your files are live.

## 7. The everyday workflow (repeat forever)

Whenever you change something:
```bash
git add .                                  # stage your changes
git commit -m "Add Grad-CAM explainability demo"   # snapshot with a clear message
git push                                   # upload to GitHub
```
That's the whole loop: **edit → add → commit → push.**

## 8. Good habits that impress reviewers

- **Commit small and often** with descriptive messages, not one giant "final" commit.
- **Keep `README.md` current** — it's the first (and sometimes only) thing people read.
- **Never commit large datasets or trained model files** (that's what `.gitignore` is for); instead
  describe in the README where to get the data.
- **Use branches for experiments:** `git checkout -b idea-fewshot` makes a parallel copy so you don't
  break `main`; merge it back when it works.
- **Add a `results/` note or table** in the README so your professor sees outcomes at a glance.
- **Tag your submission:** `git tag v1.0-presentation && git push --tags` marks the exact version you
  presented.

## 9. Optional but classy: a repository "front door"
- Add **badges** (Python version, license) at the top of the README.
- Put one clear **architecture diagram** in `figures/` and embed it in the README with
  `![architecture](figures/architecture.png)`.
- If you build notebooks, add an **"Open in Colab"** badge so anyone can run them in the browser.

## 10. Quick Git cheat-sheet

| Goal | Command |
|------|---------|
| Start tracking a folder | `git init` |
| Stage all changes | `git add .` |
| Snapshot with a message | `git commit -m "message"` |
| Link to GitHub | `git remote add origin <url>` |
| Upload | `git push` (first time: `git push -u origin main`) |
| Download others' changes | `git pull` |
| See history | `git log --oneline` |
| New experiment branch | `git checkout -b my-branch` |
| See what changed | `git status` / `git diff` |

---

You now have the full masterclass: reading order, three deep dives, the synthesis, future work with
skills, and a publishing guide. Good luck with the presentation — reread `01_reading_order.md`'s
"4-pass technique" and the "what your professor will ask" list the night before.
