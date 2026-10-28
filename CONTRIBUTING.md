# Contributing

## Branch model

```text
main  <- release/final only
  ^
dev   <- reviewed integration branches
  ^
feat/* | fix/* | chore/* | docs/* | test/*
```

1. Đồng bộ `dev` trước khi bắt đầu.
2. Tạo một nhánh mới từ `dev` cho đúng một work item.
3. Viết/chỉnh test cùng thay đổi khi bắt đầu code.
4. Chạy local quality gate và OpenCode review chỉ đọc.
5. Push nhánh công việc, tạo PR vào `dev`; không push trực tiếp vào `main`.
6. Merge `dev` vào `main` chỉ khi laptop/Pi/release gate tương ứng đã đạt.

## Tên nhánh

- `feat/lxx-short-name`
- `fix/lxx-short-name`
- `chore/lxx-short-name`
- `docs/lxx-short-name`
- `test/lxx-short-name`

## Commit

Dùng Conventional Commits: `type(scope): description`.

Ví dụ:

- `chore(env): prepare Python 3.11 workspace`
- `feat(data): add group-safe split validator`
- `test(edge): cover QR replay rejection`

## Local environment

```powershell
uv python install 3.11
uv sync --frozen
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type commit-msg
```

Nhóm `colab` không được cài trên laptop. Colab dùng requirements đã export riêng.

Project hiện đặt `tool.uv.package = false` vì chưa có application package và chưa có code chức năng. Khi tạo package đầu tiên trong `src/`, work item đó phải bật packaging/editable install và cập nhật lockfile.

Không chạy `uv sync --extra colab` trên laptop. Extra này chỉ tồn tại để khóa constraint; Colab dùng `requirements/colab.lock.txt` và runtime `torch`/`torchvision` có sẵn.

## Quality gate khi bắt đầu có code

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest -q --cov=src --cov-report=term-missing
uv run pip-audit
```

Không commit dataset, checkpoint, ONNX, firmware binary, key hoặc `.env`.

Workflow `Environment` hiện chỉ kiểm cấu hình, format, tooling type-check và dependency audit vì repo chưa có application code/test. Work item code đầu tiên phải thêm `mypy src tests` và `pytest` vào CI trước khi được merge vào `dev`.
