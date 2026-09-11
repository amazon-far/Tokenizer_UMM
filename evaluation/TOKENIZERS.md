# Third-party image tokenizers

The evaluation code studies several discrete image tokenizers. Their source
lives in subdirectories under `evaluation/` (plus `data_process/vqgan/`) and has
been **adapted/modified** from the upstream projects listed below to integrate
with this codebase. Each subdirectory carries the upstream project's `LICENSE` —
please consult and comply with each project's own license. This third-party code
is **excluded from** the repository's top-level Apache-2.0 grant.

> **No model weights are included in this repository.** Download the tokenizer
> weights yourself from the upstream sources and place them at the paths shown
> below. All weight files (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`,
> `*.jit`, ...) are git-ignored.

Directories are relative to `evaluation/` unless otherwise noted. The License
column links to the copy shipped in this repository.

| Tokenizer | Directory | Upstream project | License |
|-----------|-----------|------------------|---------|
| IBQ (SEED-Voken) | `seed_voken/` | [TencentARC/SEED-Voken](https://github.com/TencentARC/SEED-Voken) | [Apache-2.0](seed_voken/LICENSE) |
| GigaTok / GigaTok-DINO | `GigaTok/` | [SilentView/GigaTok](https://github.com/SilentView/GigaTok) | [MIT](GigaTok/LICENSE) |
| UniTok / UniTok-sem | `unitok/` | [FoundationVision/UniTok](https://github.com/FoundationVision/UniTok) | [MIT](unitok/LICENSE) |
| Chameleon (recipe sanity-check) | `chameleon/` | [facebookresearch/chameleon](https://github.com/facebookresearch/chameleon) | [Chameleon Research License](chameleon/LICENSE) |
| Chameleon VQGAN (data tokenization) | `../data_process/vqgan/` | [facebookresearch/chameleon](https://github.com/facebookresearch/chameleon) | [Chameleon Research License](../data_process/vqgan/LICENSE) |

> ⚠️ **Chameleon is noncommercial-research-only.** The code in
> `evaluation/chameleon/` and `data_process/vqgan/` is governed by the
> [Chameleon Research License](chameleon/LICENSE), which grants rights solely for
> "Noncommercial Research Uses" (§2.2.1, §3.1) and incorporates Meta's
> [Chameleon Acceptable Use Policy](https://ai.meta.com/resources/models-and-libraries/chameleon-use-policy/)
> by reference. Both directories carry a `NOTICE` file with the attribution
> required by §2.2.3. If you need a fully permissive pipeline, use one of the
> other tokenizers above — Chameleon is only used for the recipe sanity-check
> and as the default VQGAN in the data-processing scripts.

## Where to place weights

Paths below are relative to the repository root. `$DATA_ROOT` refers to the
`DATA_ROOT` entry in your `.env` (see `.env.example`).

- **IBQ (SEED-Voken)** — `evaluation/seed_voken/imagenet256_{1024,8192,16384}.ckpt`.
  Configs are provided under `evaluation/seed_voken/configs/IBQ/`.
- **GigaTok** — `evaluation/GigaTok/VQ_BL256_e200.pt` (and
  `evaluation/GigaTok/VQ_BL256_dino_disc.pt` for the DINO variant). Configs are
  provided under `evaluation/GigaTok/configs/vq/`.
- **UniTok** — `$DATA_ROOT/unitok_checkpoint_large_clip_1codebook/ckpt-last.pth`
  (and `..._sem/ckpt-last.pth` for the semantic variant).
- **Chameleon VQGAN** — `data/tokenizer/vqgan.ckpt` and `data/tokenizer/vqgan.yaml`
  (the same pair is passed as `--vqgan_path $REPO_ROOT/data/tokenizer` by the
  data-processing scripts; see [Data.md](../Data.md)).

Exact expected paths are defined in `evaluation/T2I_Eval/decode.py` (image
generation / decoding) and `evaluation/VQA_Eval/model_vqa_loader.py`
(understanding); adjust there if you store weights elsewhere.
