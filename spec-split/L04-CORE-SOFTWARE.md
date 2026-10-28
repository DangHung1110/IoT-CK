# L04 - QR, decision và edge core không phần cứng

## Mục tiêu

Hoàn thiện logic nghiệp vụ chạy thuần trên laptop, mọi I/O sau interface để có thể thay adapter Pi/ESP32 về sau.

## Module

- `shared/qr.py`: canonical payload, sign/verify HMAC.
- `edge/qr_reader.py`: decode từ ảnh, nhưng verify tách khỏi decode.
- `edge/classifier.py`: model loader, preprocess, ONNX inference.
- `edge/decision.py`: pure function kết hợp QR status + AI policy.
- `edge/camera.py`: `FolderCamera`, `OpenCVCamera`; chưa có Picamera2.
- `edge/controller.py`: `MockController`; chưa có serial thật.
- `edge/orchestrator.py`: state machine không phụ thuộc sleep thật bằng clock abstraction.

## QR contract

Payload giữ dạng `PSv1|parcel_id|warehouse|ts_utc|mac`.

- `parcel_id`: `[A-Z0-9]{6,20}`.
- warehouse thuộc allowlist.
- parse timestamp UTC nghiêm ngặt.
- dùng `hmac.compare_digest`.
- từ chối quá cũ và timestamp vượt quá future skew cho phép.
- replay store là interface để test in-memory/SQLite.
- QR malformed/invalid không bao giờ được ghi vào seen store.

## Decision order

1. `QR_UNREADABLE`.
2. `QR_INVALID`.
3. `QR_EXPIRED` hoặc future timestamp invalid.
4. `QR_REPLAY`.
5. `WAREHOUSE_UNKNOWN`.
6. `AI_ERROR`.
7. damage theo class threshold/policy.
8. uncertain.
9. `OK_A` hoặc `OK_B`.

QR fail phải short-circuit; classifier không được gọi.

## State machine laptop

`IDLE -> DETECTED -> SETTLING -> CAPTURE -> VERIFY_QR -> ANALYZE -> DECIDE -> ACTUATE -> COMMIT -> IDLE`.

Timeout/exception chuyển `ERROR`; không đánh dấu parcel đã xử lý nếu actuation chưa có DONE. Mock clock/event queue giúp test không phải chờ thời gian thật.

## Work items

### L04.1 - QR sign/verify

Property tests cho canonicalization, tamper, expiry, future skew, replay và Unicode/whitespace input.

### L04.2 - Decision pure function

Mỗi dòng decision table có test; thêm boundary test đúng bằng threshold, NaN, class thiếu và policy không hợp lệ.

### L04.3 - Classifier adapter

Nạp ONNX/card atomically, verify checksum/schema, preprocess và infer N frames. Timeout/failure trả structured error.

### L04.4 - Camera/controller mocks

FolderCamera phát fixture có metadata; MockController mô phỏng ACK/DONE/timeout/JAM/ESTOP deterministically.

### L04.5 - Orchestrator

Test happy path A/B/reject và mọi timeout. Không network, không database thật ở work item đầu.

## Acceptance tests

- Đủ mọi decision row và boundary.
- QR fail -> classifier call count bằng 0.
- ACK nhưng thiếu DONE -> ERROR, không mark seen.
- AI exception -> reject `AI_ERROR`.
- Model checksum sai -> app không khởi động với model mới.
- Shutdown giữa event không làm state corruption.

## Done

- Unit tests logic đạt coverage >= 90%.
- Không import `picamera2`, GPIO hay serial trong laptop profile.
- OpenCode review security/state-machine đạt APPROVED.
- Commit đề xuất: `feat(edge): add secure QR and hardware-free sorter core`.
