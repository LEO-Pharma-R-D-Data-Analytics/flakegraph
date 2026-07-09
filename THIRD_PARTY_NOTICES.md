# Third-Party Notices

This file summarizes important third-party license and terms considerations for
FlakeGraph. It is informational and is not a substitute for legal review.

FlakeGraph source code is licensed under the MIT License. That license applies
to this repository's application code, configuration examples, prompt templates,
documentation, tests, and original sample documents. It does not re-license
third-party packages, provider services, downloaded model weights, or operating
system packages installed into a Docker image.

## Dependency Installation

The source repository does not vendor Python packages or model weights. Runtime
dependencies are resolved from `pyproject.toml` and `uv.lock`; Docker images
install those dependencies during the image build.

If you redistribute a built image or packaged artifact, include the license and
notice files required by the installed Python packages, model artifacts, and
system packages in that artifact.

## Notable Runtime Components

- `mineru[pipeline]` is optional in Python packaging but installed by the
  default Docker build profile. MinerU is licensed under the MinerU Open Source License,
  based on Apache 2.0 with additional terms. Review the attribution obligation
  and commercial thresholds before offering MinerU-backed services.
  Source: https://github.com/opendatalab/MinerU/blob/master/LICENSE.md
- `sentence-transformers` is used by the local embedding profile. The default
  model, `sentence-transformers/all-MiniLM-L6-v2`, is published on Hugging Face
  with an Apache-2.0 license. Review the selected model card when changing
  models. Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- `torch` and `torchvision` are used by the local embedding and MinerU profiles.
  The project pins CPU wheels for Linux builds. Review PyTorch package licenses
  and notices when redistributing images.
- `snowflake-connector-python` is used for Snowflake connectivity and is
  Apache-2.0 licensed. Snowflake Cortex and Snowpark Container Services usage is
  governed by the target Snowflake account's terms and configuration.
- `tesseract-ocr` is installed only when `KG_INSTALL_TESSERACT=true`; Tesseract
  is Apache-2.0 licensed. Source: https://tesseract-ocr.github.io/tessdoc/
- `poppler-utils` is installed only when `KG_INSTALL_TESSERACT=true`; distro
  packages include GPL/LGPL/MIT licensed components. Review this optional image
  variant before redistribution.

## Provider Services And Models

FlakeGraph can call OpenAI-compatible APIs, Azure OpenAI, vLLM endpoints,
Snowflake Cortex, generic HTTP OCR services, MinerU API services, and local
models. The configured provider or model owner controls the applicable terms,
data usage policy, allowed outputs, region availability, and commercial
permissions.

Before production use, approve every configured provider endpoint and model
identifier for licensing, data residency, security, and acceptable-use posture.
