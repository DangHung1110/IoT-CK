# L05 - Storage, uplink, console MVP và replay integration

## Mục tiêu

Chạy end-to-end trên laptop bằng folder/webcam + ONNX + mock controller, lưu event bền vững và hiển thị được qua Console tối thiểu.

## Storage contract

- SQLite WAL, foreign keys bật.
- `events(event_uuid UNIQUE, ...)`, `seen_parcels`, `outbox`.
- Event, seen state và outbox enqueue phải có transaction boundary rõ ràng.
- Uplink dùng idempotency key `event_uuid`; retry không tạo bản ghi trùng.
- Outbox exponential backoff có jitter, giới hạn attempts chỉ để cảnh báo chứ không âm thầm bỏ metadata.
- Ảnh có retention riêng; khi hết dung lượng có thể bỏ ảnh cũ nhưng giữ metadata/audit.

## Console MVP trong phase này

Chỉ gồm:

- nhận `POST /api/v1/events` idempotent;
- xem danh sách/chi tiết event;
- tạo parcel + QR hợp lệ để test;
- health endpoint;
- SQLite local.

Chưa làm ở phase này: RBAC đầy đủ, 2FA, model rollout, ThingsBoard, production HTTPS, firmware release UI.

## Replay suites

### Suite A - Logic fixture

- `FakeClassifier` và QR fixture kiểm 100% expected tray/reason.
- Ít nhất 50 cases, bao phủ QR lỗi, mỗi class, uncertain, AI error, controller timeout.
- Đây là acceptance test software.

### Suite B - Real model

- ONNX chạy trên labeled replay set.
- Báo confusion matrix/damage recall/false reject/uncertain; không yêu cầu 100%.
- Đây là model evaluation, không dùng để che lỗi orchestration.

## Work items

### L05.1 - SQLite repositories/migrations

Schema versioned, transaction tests, crash/restart tests.

### L05.2 - Outbox worker

HTTP fake server trả success/timeout/500/duplicate; verify ordering, retry và idempotency.

### L05.3 - Console ingest/read

API schema versioned; validate payload size/type; safe image filename generated server-side.

### L05.4 - QR generation

Dùng chung canonical signer với edge; secret lấy từ env/file, không log và không commit.

### L05.5 - Replay harness

CLI xuất JSON + Markdown report; exit non-zero khi Suite A mismatch hoặc metric schema invalid.

## Acceptance tests

- Mất mạng khi enqueue, restart app, có mạng lại -> gửi đủ đúng một logical event.
- Server nhận cùng UUID hai lần -> một record.
- Crash giữa transaction -> không có seen/event/outbox lệch nhau.
- Path traversal/ảnh quá cỡ/payload sai schema -> từ chối.
- Suite A đạt 100%; Suite B xuất metric đúng protocol.

## Done

- End-to-end local chạy không cần Pi/ESP32.
- Outbox recovery và idempotency được chứng minh bằng integration test.
- Console hiển thị event và ảnh.
- OpenCode review transaction/API security đạt APPROVED.
- Commit đề xuất: `feat(integration): add durable local replay and console ingest`.
