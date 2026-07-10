# Pyramids repo

This repository is a fork of the original [`AEljarrat/pyramids`](https://github.com/AEljarrat/pyramids) project.

The original project provides Python scripts to segment SEM images of silicon pyramid facets and measure their properties. This fork keeps the original structure, while adding thesis-specific notebooks and modified source files for SEM-based pyramid segmentation and maximum-height analysis.

## Thesis analysis notebooks

This fork adds two thesis workflows:

- `Nano Texturing.ipynb` — workflow for nanotextured SEM images.
- `Standard Texturing.ipynb` — workflow for standard-textured SEM images.

Binder links:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JAYNUS98/pyramids/thesis-updated-analysis?urlpath=tree/Nano%20Texturing.ipynb) Nano Texturing

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JAYNUS98/pyramids/thesis-updated-analysis?urlpath=tree/Standard%20Texturing.ipynb) Standard Texturing

## Modified source files

The thesis-specific source files are located in the `pyramids/` folder:

- `source1_altered_nano.py`
- `source1_altered_standard.py`

These files include the modified SEM analysis workflow, including calibrated scalebar handling, updated watershed segmentation settings, manual marker correction, and unbordered maximum-height measurement.

## Local installation

The repository can be cloned and installed locally using:

```bash
pip install -e .
