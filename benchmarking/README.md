# Benchmarking and Evaluation

This directory contains the integration files and configuration needed to
evaluate feature-matching models on the Blender multi-object synthetic dataset
using the [Glue Factory](https://github.com/cvg/glue-factory) framework.

---

## 1. External Dependency: Glue Factory

Clone and install Glue Factory:

```bash
git clone https://github.com/cvg/glue-factory.git external/glue-factory
cd external/glue-factory
pip install -e .
```

Refer to the Glue Factory README for additional dependency details.

---

## 2. Integrating the Blender Dataset into Glue Factory

Copy the extension files into the Glue Factory source tree, preserving the
directory structure:

```
benchmarking/gluefactory_extension/gluefactory/
├── configs/
│   ├── superpoint+superglue_eval.yaml
│   ├── superpoint+lightglue_eval.yaml
│   ├── superpoint+lsd+gluestick_eval.yaml
│   └── loftr_eval.yaml
├── datasets/
│   └── blender.py                          # Dataset loader
├── eval/
│   └── blender.py                          # Evaluation pipeline
└── models/
    └── matchers/
        └── blender_dataset_matcher.py      # Per-object ground-truth matcher
```

Copy into:

```
external/glue-factory/gluefactory/
```

No modifications to the core Glue Factory code are required.

---

## 3. Dataset and Output Paths

Glue Factory resolves paths relative to its own root directory via
`gluefactory/settings.py`:

| Purpose              | Default path                                |
|----------------------|---------------------------------------------|
| Dataset              | `external/glue-factory/data/blender_dataset` |
| Evaluation results   | `external/glue-factory/outputs/results/`     |

Place (or symlink) the Blender dataset folder so that it is accessible at
`data/blender_dataset` inside the Glue Factory root.

---

## 4. Running the Evaluation

From the Glue Factory root directory:

```bash
python -m gluefactory.eval.blender --conf superpoint+superglue_eval
python -m gluefactory.eval.blender --conf superpoint+lightglue_eval
python -m gluefactory.eval.blender --conf superpoint+lsd+gluestick_eval
python -m gluefactory.eval.blender --conf loftr_eval
```

Refer to the Glue Factory documentation for model-specific options.

---

## 5. Precomputed Results and Dataset

Both the Blender dataset and the precomputed evaluation results used in this
thesis are available on Hugging Face:

👉 [imricc4rdo/benchmarking_data](https://huggingface.co/datasets/imricc4rdo/benchmarking_data/tree/main)

To reproduce the analysis notebooks, place the result folders under:

```
external/glue-factory/outputs/results/<result_folders>
```

---

## 6. Attribution

This work relies on the [Glue Factory](https://github.com/cvg/glue-factory)
framework for model evaluation.  All credit for the evaluation infrastructure
belongs to the original authors.  This repository only provides dataset
integration files and experimental scripts.