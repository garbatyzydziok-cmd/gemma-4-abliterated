# Abliteration Comparison: Baseline vs Abliterated

## Summary

| Metric | Baseline (Stock) | Abliterated | Change |
|--------|:----------------:|:-----------:|:------:|
| Capability Score | 20/20 (100%) | 50/50 (100%) | No degradation |
| Refusal Score | 20/20 (100%) | 50/50 (100%) | Maintained |
| Avg Generation Speed | 20.41 tok/s | 23.67 tok/s | +16% faster |
| Peak Memory (4-bit) | 18.7 GB | 19.1 GB | +0.4 GB |

## Key Findings

1. **Zero capability degradation** -- The abliterated model answers all 50 capability prompts (math, coding, reasoning, knowledge, creative, instruction-following, complex multi-turn) with full substantive responses.

2. **Zero refusals** -- All 50 refusal-trigger prompts answered across all categories (chemistry/biology, historical violence, security education, dark fiction, medical/pharmaceutical, controversial factual, weapons/historical, politically sensitive, mature educational).

3. **Speed improvement** -- The abliterated model is 16% faster on average (23.67 vs 20.41 tok/s). This is likely due to reduced "thinking" overhead when the model no longer evaluates safety heuristics.

4. **Baseline was already permissive** -- Stock Gemma 4 31B-IT answered all 20 initial test prompts with disclaimers but substantive content. The abliteration primarily removes residual disclaimer/hedging behavior and ensures consistent responses across edge cases.

## Methodology

- **Baseline**: Evaluated on 20 capability + 20 refusal prompts (stock mlx-community/gemma-4-31b-it-4bit)
- **Abliterated**: Evaluated on 50 capability + 50 refusal prompts (expanded test suite)
- **Hardware**: Apple M4 Max, 64GB unified memory, MLX 4-bit quantization
- **Max tokens**: 512 per prompt
- **Refusal classifier**: Dual-signal regex detection (refusal patterns + answer signals)

## Abliteration Configuration

- **Weight factor**: 1.0 (conservative)
- **Layers modified**: All 60 transformer layers
- **Weight matrices**: embed_tokens, self_attn.o_proj, mlp.down_proj
- **Strongest refusal signal layers**: 39, 40, 41, 38, 42 (middle-to-late layers)
- **Prompt datasets**: 256 harmful (AdvBench-style) + 256 harmless (alpaca-style)

## Detailed Results

See `results/baseline_results.json` and `results/abliterated_results.json` for full per-prompt responses and metrics.
