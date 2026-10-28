# L03 - Evaluation, threshold, ONNX export và benchmark laptop

## Mục tiêu

Đánh giá candidate và export ONNX trên Google Colab bằng cùng protocol; sau đó chỉ tải ONNX artifact xuống laptop để kiểm software và benchmark CPU.

## Metric

- Confusion matrix và precision/recall/F1 từng lớp.
- Damage recall: mọi class khác `ok` là damage.
- False reject: ảnh `ok` bị reject.
- Coverage/uncertain rate.
- Accuracy 5 lớp là metric phụ.
- Latency p50/p95, RSS memory, model size.

Mọi metric policy phải gọi đúng hàm decision policy dùng ở runtime.

## Policy thống nhất

Input: vector xác suất theo class names trong model card.

1. Nếu inference lỗi/NaN/sai shape -> `AI_ERROR`.
2. Nếu top class là damage và xác suất đạt `tau_damage_by_class[class]` -> `DAMAGE_<CLASS>`.
3. Nếu top class là `ok` và `p_ok >= tau_ok_pass` -> PASS.
4. Còn lại -> `AI_UNCERTAIN` và REJECT.

Threshold được chọn trên validation. Test chỉ chạy sau khi model + threshold + preprocess đã đóng băng.

## Model card bắt buộc

```json
{
  "schema_version": 1,
  "model_name": "candidate-name",
  "model_version": "v0.1.0",
  "git_commit": "...",
  "dataset_version": "v0",
  "dataset_manifest_sha256": "...",
  "classes": ["..."],
  "input": {"name": "images", "shape": [1, 3, 320, 320], "dtype": "float32"},
  "preprocess": {"id": "...", "roi": "square", "resize": "...", "color": "RGB", "scale": 255.0, "mean": [0,0,0], "std": [1,1,1], "layout": "NCHW"},
  "output": {"name": "...", "semantics": "logits-or-probabilities"},
  "thresholds": {"tau_ok_pass": 0.0, "tau_damage_by_class": {}},
  "metrics": {},
  "runtime": {"onnxruntime": "...", "opset": 0},
  "sha256": "..."
}
```

## Work items

### L03.1 - Eval engine trên Colab

Tải predictions một lần, lưu immutable prediction table; tính metric/policy từ table để dễ audit.

### L03.2 - Threshold selection trên Colab

Quét threshold trên validation với constraint damage recall trước, false reject sau. Lưu toàn bộ frontier, không chỉ điểm thắng.

### L03.3 - ONNX export/parity trên Colab

- Export fixed batch 1, opset được pin.
- So sánh PyTorch với ONNX trên ít nhất 20 ảnh gồm đủ lớp và case biên.
- Kiểm class order, output semantics, max absolute diff và decision parity.
- Decision parity phải 100%; numeric tolerance đặt theo FP32.

### L03.4 - Tải artifact và benchmark laptop

- Tải đúng bộ `model.onnx`, `model_card.json` và checksum từ Google Drive; không tải checkpoint PyTorch nếu không cần.
- Chỉ chạy ONNX Runtime inference, không train/fine-tune trên laptop.
- Warmup 10, đo ít nhất 100 inference, batch 1.
- Đo preprocess riêng, inference riêng và end-to-end.
- CPU 1/2/4 threads là bắt buộc vì Pi dùng CPU. Không benchmark/train GPU laptop.

### L03.5 - Candidate decision

Decision record ghi candidate thắng và lý do. Nếu không model nào đạt KPI trên v0 thì vẫn chọn artifact để test software, nhưng gắn `quality_status=smoke_only`.

## Acceptance tests

- Eval không đọc test khi mode là threshold tuning.
- Runtime policy và eval policy cho kết quả giống nhau trên fixture.
- Model/card checksum mismatch -> từ chối load.
- Sai class order/input shape/output NaN -> fail rõ ràng.
- ONNX/PyTorch decision parity 100% trên parity set.

## Done

- Google Drive có checkpoint + ONNX + model card + checksum + prediction/eval report; laptop chỉ có ONNX bundle cần cho test.
- Có decision record chọn candidate hoặc ghi rõ smoke-only.
- OpenCode review tập trung metric leakage và parity đạt APPROVED.
- Commit đề xuất: `feat(ml): evaluate export and benchmark ONNX candidates`.
