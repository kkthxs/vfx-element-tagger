# Third Party Components and Model Terms

The application source is MIT licensed. That does not relicense downloaded weights,
external tools, optional dependencies or media. The source distribution does not bundle
production footage or weights. This inventory is a starting point, not legal clearance
for a particular deployment or redistributable binary.

## Analysis Models

Model cards were checked on 2026-10-07. Review the files/terms at the exact revision
you download; mutable repository metadata is not a substitute for the actual license.

| Snapshot | Current stated terms | Release handling |
| --- | --- | --- |
| [Qwen3.5-4B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) | Apache-2.0 in the conversion card, inherited from Qwen | User-downloaded primary; not bundled |
| [MiniCPM-o-4_5-4bit](https://huggingface.co/mlx-community/MiniCPM-o-4_5-4bit) | Apache-2.0 in the conversion card; also stated by [the upstream model](https://huggingface.co/openbmb/MiniCPM-o-4_5#license) | Optional user download; not bundled |
| [Marlin-2B-MLX-8bit](https://huggingface.co/NemoStation/Marlin-2B-MLX-8bit) | Apache-2.0 in the conversion card; access is gated | Optional user download subject to access; not bundled |

The optional embedding snapshot and legacy PyTorch-model stack are not part of the
recommended release workflow. Their terms and runtime compatibility need separate
review before they are promoted or redistributed. Do not assume all Qwen/MiniCPM
versions have identical terms merely because these selected cards do.

The downloader performs a narrow local Marlin metadata-discriminator repair and records
it in the manifest. Preserve source model licenses and modification records if you
redistribute any repaired snapshot. No redistribution approval is inferred here.

## Runtime and External Programs

- [MLX](https://github.com/ml-explore/mlx) and [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
  are separate upstream projects. Adapted runtime-fix code in `model_backends.py` retains
  the MLX-VLM license in the root `NOTICE` file.
- FFmpeg/ffprobe are installed separately. FFmpeg's applicable license depends on its
  build: optional GPL components change the licensing obligations. Review the official
  [FFmpeg legal guidance](https://ffmpeg.org/legal.html) before distributing an installer
  or bundled executable. Invoking a system executable does not make that executable MIT.
- OpenImageIO, Pillow, NumPy, OpenCV, Transformers, Hugging Face Hub and their dependencies
  retain their own notices. This is not a complete binary-distribution SBOM.

Before a packaged binary/installer release, inventory the resolved dependency versions,
retain required licenses/notices and review codec/patent obligations as applicable.
Source-only publication and redistribution of a complete model/tool bundle have different scopes.

## Media and Evaluation Data

Do not distribute stock/production clips, posters or screenshots without permission.
Even text-only catalogs and evaluation reports can reveal confidential filenames,
source paths, reviewer notes and artist identities. The repository's test constraints
are not a grant of rights to the referenced media. Use synthetic fixtures or media
with explicit publication rights for public demonstrations and reproducible tests.
