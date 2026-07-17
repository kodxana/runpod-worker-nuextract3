# Third-Party Notices

This worker bundles third-party software and a model downloaded during the
container build. The build resolves the versions pinned in `requirements.txt`;
their own licenses and notices continue to apply.

| Component | License | Source |
| --- | --- | --- |
| NuMind NuExtract3 | Apache-2.0 | https://huggingface.co/numind/NuExtract3/tree/2e9fca82ee641e6bb6e1f5d905241e994be27a07 |
| PyTorch and torchvision | BSD-3-Clause | https://github.com/pytorch/pytorch |
| Hugging Face Transformers | Apache-2.0 | https://github.com/huggingface/transformers |
| Hugging Face Hub | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| Runpod Python SDK | MIT | https://github.com/runpod/runpod-python |
| Flash Linear Attention | MIT | https://github.com/fla-org/flash-linear-attention |
| causal-conv1d | BSD-3-Clause | https://github.com/Dao-AILab/causal-conv1d |
| Pillow | HPND | https://github.com/python-pillow/Pillow |
| pypdfium2 5.12.0 | BSD-3-Clause and Apache-2.0 | https://github.com/pypdfium2-team/pypdfium2 |
| PDFium | BSD-style license and bundled dependency licenses | https://pdfium.googlesource.com/pdfium/ |
| httpx | BSD-3-Clause | https://github.com/encode/httpx |
| jsonschema | MIT | https://github.com/python-jsonschema/jsonschema |
| boto3 | Apache-2.0 | https://github.com/boto/boto3 |

The base image is `docker.io/pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`, pinned by
digest in `Dockerfile`. NVIDIA CUDA and cuDNN components retain their upstream
license terms.

The pinned pypdfium2 wheel bundles PDFium and its applicable license files. Those
files remain installed in the Python distribution metadata inside the image.

The image contains NuExtract3 only at revision
`2e9fca82ee641e6bb6e1f5d905241e994be27a07`. The build verifies all LFS-backed
runtime artifacts by size and SHA-256 and retains the snapshot's Apache License
2.0 file under `/opt/models/nuextract3`. Repository metadata and model-card front
matter both declare `apache-2.0`. Apache-2.0 permits commercial use and
redistribution subject to its notice, attribution, patent, and other license
conditions.
