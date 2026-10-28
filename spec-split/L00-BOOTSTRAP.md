# L00 - Bootstrap repo và môi trường laptop

## Mục tiêu

Tạo repo tái lập được trên Windows laptop và tương thích Python 3.11 của Raspberry Pi về sau.

## Hiện trạng đã kiểm tra

- Git 2.47.1: có.
- Docker 28.1.1: có; cần kiểm daemon khi đến phase integration.
- OpenCode 1.18.35: có; hiện chưa đọc được thư mục log trong sandbox, cần chạy ngoài sandbox/quyền người dùng bình thường.
- `uv` 0.12.15: có.
- GPU: NVIDIA GeForce RTX 3050 Laptop 6 GB, nhưng theo quyết định dự án **không dùng để train** nhằm tránh tải nhiệt và ảnh hưởng máy.
- Python project chuẩn 3.11: chưa có. `python` hiện trỏ tới MSYS2 Python 3.12.7, không dùng cho repo.
- `pre-commit`, `gitleaks`, GitHub CLI: chưa có; gitleaks/gh là tùy chọn ở L00, pre-commit nên có.

## Không cần cài

- Không cài CUDA Toolkit, PyTorch CUDA, TensorRT, WSL hoặc Docker GPU trên laptop.
- Laptop chỉ cần ONNX Runtime CPU để test inference giống hướng triển khai Pi.
- Không cần plugin/skill AI bổ sung để bắt đầu. GitHub plugin chỉ hữu ích sau khi có remote/PR. Google Drive là nơi lưu dữ liệu/checkpoint/model; Colab mount Drive để train.

## Cấu trúc repo ban đầu

```text
parcel-sorter/
  pyproject.toml
  uv.lock
  .python-version
  .gitignore
  .pre-commit-config.yaml
  README.md
  specs/
  src/parcel_sorter/
    ml/
    edge/
    shared/
    console/
  tests/
    unit/
    integration/
    fixtures/
  configs/
  scripts/
  data/README.md
  artifacts/README.md
```

## Dependency groups

- `core`: pydantic, pydantic-settings, PyYAML, structlog, typer, rich.
- `vision-runtime`: numpy, Pillow, opencv-python, onnxruntime.
- `ml-colab`: torch, torchvision, ultralytics, pandas, scikit-learn, matplotlib; chỉ cài trong Colab, không sync mặc định trên laptop.
- `qr`: qrcode, pyzbar.
- `console`: FastAPI, uvicorn, SQLModel, Jinja2, python-multipart, httpx.
- `dev`: pytest, pytest-cov, hypothesis, ruff, mypy, pre-commit, pip-audit.

Chưa chốt số phiên bản bằng văn bản. Khi khởi tạo repo, dùng resolver để tạo `uv.lock`; export riêng dependency train cho Colab. Không nâng phiên bản giữa một experiment series.

## Work items

### L00.1 - Git skeleton

- Khởi tạo Git, `.gitignore`, README, license cho code của nhóm.
- Copy bộ spec đã tách vào `specs/`.
- Tạo placeholder package và một smoke test.

### L00.2 - Python/uv

- `uv python install 3.11` và pin `.python-version` là 3.11.
- Tạo `pyproject.toml`, dependency groups và `uv.lock`.
- Xác nhận Python đến từ `.venv`, không phải MSYS2.

### L00.3 - Quality/security

- Cấu hình ruff, mypy, pytest, coverage và pre-commit.
- Thêm secret scan. Ưu tiên gitleaks nếu cài binary được; nếu chưa thì dùng detect-secrets trong pre-commit và bổ sung gitleaks ở CI sau.
- Tạo CI chạy CPU-only. Training GPU chỉ chạy trên Google Colab, không có job GPU local.

### L00.4 - OpenCode review proof

- Chạy một review read-only trên diff bootstrap.
- Lưu output review và sửa findings trước commit.

## Acceptance tests

```powershell
uv run python -c "import sys; assert sys.version_info[:2] == (3, 11)"
uv sync --frozen --group dev --group vision-runtime --group qr --group console
uv run pytest -q
uv run ruff check .
uv run mypy src tests
```

## Done

- Clone sạch có thể `uv sync --frozen` và chạy test.
- Không có file máy cá nhân, secret, dataset/model binary trong Git.
- OpenCode review bootstrap đạt APPROVED.
- Commit đề xuất: `chore(repo): bootstrap reproducible Python 3.11 project`.
