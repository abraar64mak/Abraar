# Concept Transformer on CUB-200-2011: how to reproduce

This note goes with the notebook `concept_transformer_colab_A100Final.ipynb`. The notebook sets up, trains and evaluates IBM's
Concept Transformer (<https://github.com/IBM/concept_transformer>, Rigotti et al., ICLR 2022) on the CUB-200-2011 bird dataset,
then looks at the concept-based explanations it produces. All outputs are already saved in the notebook.

## Result in one line

87.1% test accuracy (5,049 of 5,794 images) after 50 epochs, about 37 minutes of training on an A100.

## What you need

- A Google Colab runtime with a GPU (I used an **NVIDIA A100 40 GB**). The notebook works on a smaller GPU, such as a T4, but is much slower.
- About 5 GB of free disk space (the dataset is about 1.1 GB, plus cropped copies).
- Internet access, because the notebook downloads the repo, the packages, and the dataset.
- Optional: Google Drive, used at the end to save the logs and checkpoint.

## Steps to reproduce

1. Open Colab and upload the notebook (`File → Upload notebook`).
2. Choose `Runtime → Change runtime type` and select an **A100 GPU** (a T4 also works).
3. Run `Runtime → Run all`. Click "Allow" when Colab asks for Drive access near the end.
4. Wait about 10 to 15 minutes for setup (clone, install, download, one-off image cropping), then about 37 minutes for training and testing.

No manual steps are needed. For a quick check that everything works, set `DEBUG = True` in the first code cell and run cells 1, 11 and 12. This runs one batch in a minute or two. Set it back to `False` for the real run.

## Where things are (all paths are set in the first code cell)

| Variable | Default | Meaning |
|---|---|---|
| `REPO_DIR` | `/content/concept_transformer` | Where IBM's repo is cloned |
| `DATA_DIR` | `/content/data/cub2011` | Passed to `--data_dir`. Images end up in `DATA_DIR/CUB_200_2011/images` |
| `DRIVE_OUT_DIR` | `/content/drive/MyDrive/concept_transformer_run` | Where results are copied at the end |

The notebook downloads the dataset from the official source,
<https://www.vision.caltech.edu/datasets/cub_200_2011/> (archive hosted at <https://data.caltech.edu/records/65de6-vp158>).
If that download fails, download `CUB_200_2011.tgz` manually and place it in `DATA_DIR`. The notebook then unpacks it.
The dataset and checkpoints are **not** included with this submission.

## Environment used

| Item | Version |
|---|---|
| OS | Linux (Colab VM) |
| Python | 3.13.15 |
| PyTorch / torchvision | 2.11.0+cu130 / 0.26.0+cu130 (Colab preinstalled) |
| pytorch-lightning | 1.9.5 |
| torchmetrics | 0.11.4 |
| timm | 0.4.12 |
| albumentations | 1.4.24 |
| numpy / pandas | 2.1.3 / 2.2.3 (Colab preinstalled) |

Colab's preinstalled versions can change over time, so a later run may show different numbers for these. The notebook installs only
Lightning, torchmetrics, timm, albumentations and tensorboard.

I used Colab instead of a local virtual environment. A local run should work with a CUDA-enabled PyTorch plus the same `pip`
packages above, but I did not test that.

## Why the repo needed patching

The repo pins 2021 packages (`torch 1.10`, `pytorch-lightning 1.4.8`, and so on) that have no builds for current Python. I kept
Colab's PyTorch and applied small compatibility patches in the notebook's patch cell (the model code is untouched):

- Lightning's built-in fp16 instead of NVIDIA Apex, removed trainer arguments, an inline learning-rate schedule instead of `lightning-bolts`
- a NumPy 2 alias fix, the newer `accuracy()` call, and a `trainer.test()` fix
- updated albumentations imports and a Pillow function name
- the official `image_attribute_labels.txt` has a few malformed lines, which the notebook rewrites (original kept as `.orig`)

The notebook's section 2 lists each fix in a table.

## Settings used

The repo's defaults: **50 epochs, learning rate 5e-5, warm-up 10**, with batch size **64** (the repo default is 32). These are set in the first code cell.

## Outputs the notebook produces

- `train.log`: the full training log
- `logs/version_0/`: TensorBoard logs (used for the loss and accuracy curves)
- `cub_cvit/CUB2011Parts_expl1.0/binary_mnist_best_ckpt.ckpt`: the best checkpoint (lowest validation loss)
- `outputs/training_curves.png`: the training curves

## Caveats

- **Not an exact replication:** batch size 64, fp16 instead of Apex, a rewritten learning-rate schedule, and newer library versions.
- **One run, one seed.** I did not repeat it, so I can't give error bars.
- **The validation set** (1,000 images) is taken from the test set by the repo's code.
- **Explanations:** I judged them from about six species. Coarse concepts looked sensible, but the part-level ones were often generic, and the explanation loss was about 0.25 on training data but about 1.25 on validation data. I don't know why.
- A first attempt on a free T4 (5 epochs, batch 32) reached only about 9% accuracy. That run is not in the notebook.

## Troubleshooting

- `ImportError ... 'data'` in the dataset cell: restart the session and run the cells in order from the top, without skipping the clone cell.
- "No logs found" in the curves cell: the session may have restarted. Restore `logs` from Drive or re-run training.
- Re-running training deletes `logs` and `cub_cvit` first, so save to Drive before experimenting.
