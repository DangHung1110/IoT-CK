# IoT-CK

Hệ thống phân loại kiện hàng thông minh sử dụng QR, AI kiểm tra hư hại bao bì và các lớp bảo mật dành cho đồ án IoT cuối kỳ.

## Trạng thái

Môi trường phát triển, CI và hợp đồng dữ liệu L01.1 đã sẵn sàng. Đang chuẩn bị
Google Colab và Google Drive trước khi tạo pipeline dữ liệu hoặc train model.

## Nguyên tắc môi trường

- Laptop: phát triển phần mềm, unit/integration test, ONNX Runtime CPU inference và benchmark nhẹ.
- Google Drive: nguồn lưu trữ dataset, checkpoint, model export và báo cáo thí nghiệm.
- Google Colab: toàn bộ training, validation tuning và export PyTorch sang ONNX.
- Raspberry Pi/ESP32/phần cứng: chỉ bắt đầu sau khi laptop gate hoàn tất.

## Colab bootstrap

- Chạy [notebook bootstrap](notebooks/00_colab_drive_bootstrap.ipynb) bằng GPU runtime trên Google Colab.
- Notebook chỉ kiểm tra môi trường, mount Drive và ghi `colab_environment.json`; không train model.
- Bootstrap giữ nguyên scientific stack có sẵn của Colab; dependency training chỉ được cài ở notebook training riêng.
- Không cài hoặc chạy PyTorch/Ultralytics trên laptop.

## Inference ảnh và webcam trên laptop

1. Chạy [notebook export ONNX](notebooks/03_colab_export_mobilenet_onnx.ipynb) trên Colab.
   Notebook chỉ xuất model MobileNet đã thắng, kiểm tra checksum và parity; không train lại.
2. Tải file ZIP được notebook tạo trong `MyDrive/parcel_sorter/models/exports/`, rồi giải nén
   vào `models/binary-v0/`. Thư mục phải chứa `model.onnx` và `manifest.json`.
3. Cài runtime CPU nhẹ bằng `uv sync --group vision-runtime`.

Test một ảnh và lưu ảnh đã gắn kết quả:

```powershell
uv run python scripts/infer.py image --bundle models/binary-v0 path/to/parcel.jpg `
  --output artifacts/inference.jpg --show
```

Test webcam, nhấn `Q` hoặc `Esc` để thoát:

```powershell
uv run python scripts/infer.py webcam --bundle models/binary-v0 --camera 0
```

Webcam chỉ inference vùng ROI vuông cố định ở giữa khung hình, làm mượt xác suất qua 5 frame
và cần 3 quyết định liên tiếp trước khi xác nhận. Đặt một hộp lấp phần lớn ROI; nhấn `R` để
xóa lịch sử khi đổi hộp. Có thể chỉnh bằng `--roi-scale 0.7`, `--smoothing-window 5` và
`--stable-frames 3`. ROI này hỗ trợ căn camera, không phải bounding box do model tự phát hiện.

Runtime xác minh SHA-256 của ONNX trước khi nạp và áp dụng đúng threshold đã đóng băng:
`tau_normal_pass=0.95`, `tau_damage_confirm=0.50`. Model, ảnh kết quả và dữ liệu cá nhân đều
nằm trong các thư mục đã ignore, không được push lên Git.

## Git flow

- `main`: chỉ chứa phiên bản cuối đã nghiệm thu.
- `dev`: nhánh tích hợp ổn định.
- Mọi công việc thực hiện trên `feat/*`, `fix/*`, `chore/*`, `docs/*` tách từ `dev`.
- Mỗi nhánh phải qua kiểm thử, review và CI trước khi merge vào `dev`.
- Chỉ merge `dev` vào `main` khi hoàn tất một mốc phát hành.

Tài liệu đặc tả, kế hoạch triển khai và biên bản review được quản lý cục bộ, không đưa lên Git.
