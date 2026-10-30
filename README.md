# IoT-CK

Hệ thống phân loại kiện hàng thông minh sử dụng QR, AI kiểm tra hư hại bao bì và các lớp bảo mật dành cho đồ án IoT cuối kỳ.

## Trạng thái

Môi trường phát triển và CI đã sẵn sàng. Đang triển khai L01: hợp đồng dữ liệu và
kiểm tra chống rò rỉ giữa các tập train/validation/test.

## Nguyên tắc môi trường

- Laptop: phát triển phần mềm, unit/integration test, ONNX Runtime CPU inference và benchmark nhẹ.
- Google Drive: nguồn lưu trữ dataset, checkpoint, model export và báo cáo thí nghiệm.
- Google Colab: toàn bộ training, validation tuning và export PyTorch sang ONNX.
- Raspberry Pi/ESP32/phần cứng: chỉ bắt đầu sau khi laptop gate hoàn tất.

## Git flow

- `main`: chỉ chứa phiên bản cuối đã nghiệm thu.
- `dev`: nhánh tích hợp ổn định.
- Mọi công việc thực hiện trên `feat/*`, `fix/*`, `chore/*`, `docs/*` tách từ `dev`.
- Mỗi nhánh phải qua kiểm thử, review và CI trước khi merge vào `dev`.
- Chỉ merge `dev` vào `main` khi hoàn tất một mốc phát hành.

Tài liệu đặc tả, kế hoạch triển khai và biên bản review được quản lý cục bộ, không đưa lên Git.
