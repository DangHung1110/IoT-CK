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
- Không cài hoặc chạy PyTorch/Ultralytics trên laptop.

## Git flow

- `main`: chỉ chứa phiên bản cuối đã nghiệm thu.
- `dev`: nhánh tích hợp ổn định.
- Mọi công việc thực hiện trên `feat/*`, `fix/*`, `chore/*`, `docs/*` tách từ `dev`.
- Mỗi nhánh phải qua kiểm thử, review và CI trước khi merge vào `dev`.
- Chỉ merge `dev` vào `main` khi hoàn tất một mốc phát hành.

Tài liệu đặc tả, kế hoạch triển khai và biên bản review được quản lý cục bộ, không đưa lên Git.
