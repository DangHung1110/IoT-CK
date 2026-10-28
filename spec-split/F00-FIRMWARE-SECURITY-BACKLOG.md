# F00 - Backlog firmware và Secure OTA cộng điểm

Trạng thái: BACKLOG, chỉ thiết kế trong phạm vi hiện tại.

## Mục tiêu tương lai

Firmware ESP32 không chỉ “nạp qua app” mà có chuỗi tin cậy có thể demo:

- firmware có chữ ký;
- tải qua HTTPS và kiểm CA;
- kiểm hash/size/device/version;
- từ chối khóa ký sai và binary bị sửa;
- dual OTA slots + rollback sau self-test thất bại;
- chống downgrade;
- audit lần phát hành;
- Secure Boot v2/Flash Encryption trên board thử nếu chip hỗ trợ.

## Phân cấp để tránh làm quá sớm

### F01 - Firmware logic qua USB

UART parser, CRC, SEQ, whitelist, state machine, watchdog; chưa OTA.

### F02 - OTA transport an toàn

ESP HTTPS OTA, CA pinning/trust store, `ota_0`/`ota_1`/`otadata`, download timeout và error codes.

### F03 - Signed release

Ưu tiên cơ chế signed app của ESP-IDF. Desktop release tool quản lý chọn binary, version, signing/release và audit; không thay thế verification của bootloader/framework.

### F04 - Rollback/self-test

App mới kiểm UART, sensor và boot health rồi gọi `esp_ota_mark_app_valid_cancel_rollback()`. Không xác nhận thì rollback.

### F05 - Anti-rollback

Demo đầu có monotonic version trong NVS. Bản cộng điểm dùng `secure_version`/eFuse theo ESP-IDF sau khi test kỹ vì eFuse không hoàn tác.

### F06 - Secure Boot v2 + Flash Encryption

Chỉ trên ESP32 revision >= 3.0 và board thử. Sao lưu signing keys offline. Bắt đầu development mode; release mode sau khi toàn bộ OTA recovery đã chứng minh.

## Security test matrix

- Binary hợp lệ, version tăng -> update thành công.
- Sửa một byte -> từ chối.
- Ký bằng khóa khác -> từ chối.
- Server certificate sai -> từ chối.
- Version thấp hơn -> từ chối.
- Mất điện lúc download -> boot bản cũ.
- App mới boot lỗi/self-test lỗi -> rollback.
- OTA khi conveyor không READY -> từ chối.
- Private key xuất hiện trong repo/log/package -> release gate fail.

## Cảnh báo

- Secure Boot v2 trên ESP32 gốc yêu cầu chip revision 3.0 trở lên.
- Secure Boot và Flash Encryption tác động eFuse, có thể làm mất đường cứu hộ/JTAG/UART nếu cấu hình sai.
- App-level ECDSA manifest trong spec gốc có thể dùng cho demo, nhưng không nên tạo một cơ chế song song nếu signed-app verification của ESP-IDF đã đáp ứng. Chọn một chain of trust, ghi ADR và kiểm bằng tài liệu đúng phiên bản ESP-IDF.

## Evidence để được tính điểm cộng

- Video/demo bốn ca: success, tampered binary, wrong key, rollback.
- Log có mã lỗi nhưng không lộ key.
- Sơ đồ trust chain và threat model.
- Firmware release manifest/checksum/audit.
- Bằng chứng Secure Boot/eFuse trên board thử nếu thực hiện.
