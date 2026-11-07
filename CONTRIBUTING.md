# Contributing

Thanks for your interest! This project aims to stay small and readable — a good
portfolio entry rather than a product. PRs are welcome, especially:

- Additional dashboard panels or charts
- Support for other detection backbones (YOLOv9/v10, RT-DETR, …)
- Better camera backend handling on Linux/macOS
- Docs improvements and screenshots

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before you push

```bash
ruff check src tests examples
python tests/test_stats.py
```

Keep commits focused and messages in imperative mood ("Add FPS sparkline",
not "Added …" or "Adds …").
