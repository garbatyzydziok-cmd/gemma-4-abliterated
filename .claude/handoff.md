# Session Handoff

**Date:** 2026-04-03 16:00
**Session:** Abliterating Gemma 4 31B — remove guardrails while preserving quality
**Context usage at handoff:** Very high (long session with multiple iterations)

---

## What Was Accomplished This Session

- Researched Gemma 4 vs GLM-5.1 vs Qwen 3.6 vs Kimi K2.5 — full competitive analysis
- Downloaded and tested stock Gemma 4 31B on Apple Silicon M4 Max 64GB (23 tok/s, 18.7GB)
- Built abliteration pipeline (`abliterate.py`) — collects activations, computes refusal directions, modifies weights
- Built 100-prompt test suite (`test_suite.py`) — 50 capability + 50 refusal prompts
- Created GitHub repo: https://github.com/199-biotechnologies/gemma-4-abliterated
- Uploaded (broken) model to HuggingFace: dboris/gemma-4-31b-abliterated-mlx-4bit
- **Ran 3 major abliteration attempts, all failed to improve over baseline**

## Key Findings — CRITICAL FOR NEXT SESSION

### 1. Stock Gemma 4 31B is ALREADY very permissive
- Our test says 20/20 refusal prompts "answered" (with disclaimers)
- Heretic's test says stock model refuses 99/100
- **The difference is in the classifier.** Our test counts "answer with disclaimer" as answered. Heretic counts it as refused.
- The real goal: remove DISCLAIMERS, not hard refusals (there are almost none)

### 2. Three failed abliteration approaches

**v1: Simple orthogonal projection (weight=1.0, all 60 layers, embed+o_proj+down_proj)**
- RESULT: 99/100 responses degenerated into "and and and and..." repetition loops
- CAUSE: Modified 360 weight matrices at full strength, no norm preservation, broke LayerNorm expectations
- Lesson: Simple M_new = M - (M·d)⊗d without norm preservation destroys the model

**v2: Very gentle (weight=0.1, top 5 layers, o_proj+down_proj, no embed)**
- RESULT: 10/10 quality (no degeneration), but identical to stock model — abliteration had no effect
- CAUSE: Too conservative, barely touched the model
- Lesson: Weight 0.1 on 5 layers is insufficient

**v3: Norm-preserving + whitened SVD + Gaussian weights (scale 1.0-2.0, o_proj and/or down_proj)**
- RESULT: All 5 configs performed SAME or WORSE than baseline (more disclaimers!)
- CAUSE: The refusal direction appears to be WRONG or INVERTED
- Raw data:
  - Baseline: 3/7 clean, 4 disclaimers
  - scale=1.0 o_proj: 3/7 clean, 4 disclaimers (no change)
  - scale=1.5 o_proj: 2/7 clean, 5 disclaimers (WORSE)
  - scale=2.0 o_proj: 1/7 clean, 6 disclaimers (WORSE)
  - scale=1.5 o_proj+down_proj: 1/7 clean, 6 disclaimers (WORSE)
  - scale=2.0 o_proj+down_proj: 2/7 clean, 5 disclaimers (WORSE)
- Lesson: Projecting out our computed direction INCREASES cautiousness. Direction may be inverted, or the whitened SVD computed from 4-bit activations is unreliable.

### 3. What the community has done successfully

**Heretic (coder3101/gemma-4-31B-it-heretic):**
- Method: ARA (Arbitrary-Rank Ablation) with row-norm preservation
- Layers 1-59, auto-tuned by Optuna
- Result: 15/100 refusals (down from 99), KL divergence 0.0434
- Key params: preserve_good_behavior_weight=0.84, steer_bad_behavior_weight=0.0002
- **ARA does NOT use refusal directions at all** — it captures input/output tensors and optimizes matrices directly

**Heretic v2 (MPOA method):**
- Method: Magnitude-Preserving Orthogonal Ablation
- Result: 37/100 refusals, KL divergence 0.3801
- Targets o_proj (weights 1.33-1.47) and down_proj (weights 1.23-1.32)
- Peak at layer 57 for o_proj, layer 49 for down_proj

**Key insight: Heretic's ARA doesn't compute refusal directions from activation differences. It directly optimizes weight matrices using input/output tensor pairs. This is fundamentally different from our mean-difference approach.**

### 4. Why our refusal direction may be wrong

Possible causes:
1. **4-bit quantized activations are noisy** — quantization adds noise that corrupts the direction vector
2. **Whitened SVD peaked at layer 27** — but Heretic v2 MPOA peaked at layers 49-57. Our peak is way too early.
3. **The Gaussian taper centered on layer 27 means layers 49-57 get near-zero weight** — we're modifying the wrong layers entirely
4. **Mean difference may not capture refusal in Gemma 4** — the model's refusal mechanism may not be a simple linear direction (it's based on Gemini 3, a different architecture than Llama/Qwen)

## Current State

- **Branch:** `main`
- **Last commit:** `365831f Add v3 abliteration: norm-preserving + whitened SVD + Gaussian weights`
- **Uncommitted changes:** `abliterate_v3.py` (dead code fix), `results/v3_optimization.json`
- **HuggingFace model:** dboris/gemma-4-31b-abliterated-mlx-4bit — BROKEN (v1 with degeneration). Needs to be replaced or deleted.
- **bf16 weights cached:** `~/.cache/huggingface/hub/models--google--gemma-4-31b-it/` (58GB, still available)
- **Disk space:** ~90GB free

## What to Do Next

1. Read this handoff document
2. **Decide on approach:**

   **Option A: Use Heretic directly**
   - `pip install heretic-llm` — but requires CUDA (doesn't work on Apple Silicon MPS)
   - Could use Google Colab or a cloud GPU
   - Already proven to work: 15/100 refusals, 0.0434 KL divergence
   - Heretic's ARA method is fundamentally different (matrix optimization, not direction projection)
   
   **Option B: Fix our refusal direction computation**
   - Collect activations from bf16 model (not 4-bit) for cleaner direction
   - Use Heretic v2 MPOA's peak layers (49-57 for o_proj, not 27)
   - Try INVERTING the direction (project OUT the harmless direction instead of harmful)
   - Try using raw mean difference instead of whitened SVD (the whitening may have corrupted it)
   
   **Option C: Sumandora's runtime hook approach**
   - Insert ablation layers at inference time (no weight modification)
   - Completely reversible, can iterate in seconds
   - But needs adaptation for MLX (Sumandora is PyTorch-only)
   
   **Option D: Fine-tune instead of abliterate**
   - DPO/RLHF on a small dataset of "answer without disclaimers" examples
   - More work but more reliable than mechanical weight editing

3. Fix the test suite classifier to count disclaimers as refusals (align with Heretic's methodology)
4. Update HuggingFace model card to remove broken model or replace with working one

## Files to Review First

1. `/Users/biobook/Code/gemma4/results/v3_optimization.json` — full v3 test results
2. `/Users/biobook/Code/gemma4/abliterate_v3.py` — current best abliteration code
3. `/Users/biobook/Code/gemma4/test_suite.py` — 100-prompt evaluation suite
4. `/Users/biobook/Code/gemma4/TECHNICAL.md` — technical writeup (needs updating)
5. `/Users/biobook/Code/gemma4/.abliterate_checkpoints/` — cached activations and refusal directions

## Gotchas & Warnings

- **DO NOT use weight >= 1.0 with simple orthogonal projection on all layers** — it destroys the model (99/100 degeneration)
- **Python stdout buffers when run in background** — always use `PYTHONUNBUFFERED=1` prefix
- **mlx_lm.convert does NOT support gemma4** — must use `mlx_vlm.convert` instead
- **numpy does NOT support bfloat16** — must use `safetensors.torch` (framework="pt"), not safetensors.numpy
- **o_proj weight shape is (5376, 8192)** — refusal dir (5376) is in dim 0 (output space), not dim 1
- **embed_tokens weight shape is (262144, 5376)** — refusal dir is in dim 1 (hidden size)
- **The test suite regex classifier is too lenient** — counts "answer with disclaimer" as answered. Heretic counts it as refused. Need stricter classifier.
- **Gemma 4 has hybrid attention (full + sliding) and per-layer input embeddings** — architectural details that may affect which layers matter
- **Our whitened SVD peaked at layer 27, but Heretic MPOA peaked at layers 49-57** — suggests our direction extraction is wrong
- **Increasing abliteration scale INCREASED disclaimers** — strong evidence the direction is wrong or inverted
- **HuggingFace 199-biotechnologies org doesn't have write permissions** — model uploaded to dboris/ personal account instead
- **Codex CLI auth is broken** (OpenRouter 401 errors) — couldn't get GPT-5.4 review during this session
