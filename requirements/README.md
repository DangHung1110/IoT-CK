# Dependency exports

- `colab.lock.txt`: dependency export dành riêng cho Google Colab.
- Laptop dùng trực tiếp `uv.lock` và các default dependency groups trong `pyproject.toml`.

Không cài `colab.lock.txt` trên laptop.

Colab dùng bản `torch` và `torchvision` có sẵn trong runtime để giữ tương thích GPU/CUDA. Hai package này có trong `uv.lock` để ghi nhận constraint nhưng được loại khỏi file export. Notebook phải in và lưu phiên bản thực tế của Python, GPU, CUDA, torch và torchvision vào metadata mỗi run trước khi train.

Lệnh tái tạo file export:

```powershell
uv export --extra colab --no-default-groups --no-dev --no-emit-project `
  --no-hashes --frozen --prune torch --prune torchvision `
  --output-file requirements/colab.lock.txt
```
