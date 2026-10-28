# L06 - Cổng hoàn thành laptop

## Mục tiêu

Đóng băng một bản software/model candidate đủ sạch để sau này chuyển sang Pi mà không mang theo lỗi logic cơ bản.

## Gate bắt buộc

### Reproducibility

- [ ] Repo sạch clone được, `uv sync --frozen` thành công với Python 3.11.
- [ ] Tất cả config có schema/version.
- [ ] Dataset manifest, model, model card và reports có checksum.
- [ ] Không có secret/raw dataset/model binary trong Git.
- [ ] Raw/derived data, checkpoint và report gốc nằm trên Google Drive.
- [ ] Không cài hoặc chạy training stack trên laptop; laptop chỉ test ONNX Runtime/software.

### Data/model

- [ ] Không leakage theo `box_id` và hash.
- [ ] Hai candidate đã được train/eval trên Colab cùng protocol hoặc có ADR giải thích candidate bị loại sớm.
- [ ] Threshold chỉ chọn trên validation.
- [ ] ONNX/PyTorch decision parity 100% trên parity set.
- [ ] Có benchmark CPU laptop p50/p95.
- [ ] v0 được gắn `smoke_only`; không tuyên bố đạt KPI cuối nếu chưa có v1.

### Core software

- [ ] Decision table và boundary tests xanh.
- [ ] QR tamper/expiry/future/replay tests xanh.
- [ ] QR fail short-circuit AI.
- [ ] Model/card mismatch bị từ chối.
- [ ] ACK/DONE timeout không mark seen.

### Integration

- [ ] Logic replay suite >= 50 cases đạt 100%.
- [ ] Real-model replay report được sinh.
- [ ] Outbox offline/reconnect/restart/idempotency tests xanh.
- [ ] Console ingest/list/detail/QR generation chạy local.
- [ ] Soak mock 100 parcels không treo, không leak rõ rệt, không duplicate.

### Review/release

- [ ] Mỗi phase có OpenCode review artifact.
- [ ] Không còn P0/P1; P2 có disposition.
- [ ] Full quality gate xanh.
- [ ] Tag pre-Pi candidate, release note ghi known limitations.

## Command evidence

Tạo `artifacts/gates/laptop-gate-YYYYMMDD/` chứa:

- versions và system summary;
- test/coverage/ruff/mypy output;
- dataset validation report;
- model eval/parity/benchmark report;
- replay/outbox/soak report;
- OpenCode review summaries;
- SHA-256 manifest của toàn bộ evidence.

## Điều kiện chuyển Pi

Qua L06 không có nghĩa model đạt chất lượng cuối. Nó chỉ xác nhận software contract ổn. Chỉ bắt đầu Pi khi có hardware và người dùng chủ động mở scope tiếp theo.

Commit/tag đề xuất:

- `chore(release): pass laptop integration gate`
- tag `laptop-v0.1.0`
