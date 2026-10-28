# L01 - Data contract, thu thập và split

## Mục tiêu

Tạo pipeline dữ liệu trên Google Drive có provenance, không leakage và tái tạo được. v0 phục vụ plumbing; v1 phục vụ KPI cuối. Repo Git không chứa raw dataset.

## Google Drive là nguồn dữ liệu chính

```text
MyDrive/parcel_sorter/
  data/
    raw/<session_id>/
    derived/<dataset_version>/
    external/<source_name>/
    feedback/
    manifests/dataset_vN.csv
    archives/dataset_vN_splits.zip
  models/
    runs/<run_id>/
    exports/<model_version>/
  reports/<run_id>/
```

- Ảnh gốc, ảnh derived, archive split, checkpoint và model artifact lưu trên Drive.
- Laptop chỉ giữ fixture nhỏ đã ẩn thông tin cần thiết cho unit/integration test.
- Nếu chụp/import ảnh bằng laptop, output phải ghi thẳng vào thư mục Google Drive đồng bộ hoặc upload vào Drive trước khi tạo dataset version.
- Colab mount Drive, kiểm checksum, rồi copy archive sang `/content` để train; không train trực tiếp trên mounted Drive vì I/O chậm.

## Nhãn

| Class | Quy tắc |
| --- | --- |
| `ok` | Nguyên vẹn; chứa hard negatives như nhăn nhẹ, nhãn, chữ in, bóng đổ |
| `rach` | Rách/xé/bung băng keo nhưng chưa thủng lộ hàng |
| `mop` | Biến dạng hình học, chưa rách |
| `vo` | Thủng/nát/sập/lộ hàng |
| `uot` | Ướt/ố/bẩn nặng làm thay đổi trạng thái bao bì |

Ảnh nhiều lỗi có `primary_class` theo rule đã thống nhất để train single-label, đồng thời ghi `observed_defects` là danh sách đầy đủ. Không đổi rule giữa dataset version.

## Manifest bắt buộc

Mỗi ảnh có: `image_id`, `raw_path`, `derived_path`, `sha256_raw`, `sha256_derived`, `box_id`, `primary_class`, `observed_defects`, `source`, `source_detail`, `license`, `session_id`, `camera_id`, `captured_at_utc`, `capture_params`, `labeler`, `reviewed_by`, `split`, `dataset_version`.

Raw là bất biến. Derived ROI được tạo bằng script có version/config. Không sửa ảnh trực tiếp bằng tay.

## Split contract

- Đơn vị split là `box_id`.
- Một `box_id` chỉ thuộc đúng một split ở mọi trạng thái hư hại.
- Test dùng box mới và session giữ riêng; không dùng để chọn transform, model, hyperparameter hoặc threshold.
- External/feedback chỉ được vào train ở v1 đầu tiên; muốn đưa feedback vào val/test phải tạo protocol mới và dataset version mới.
- Script dừng với lỗi nếu có group leakage, file trùng hash giữa splits, class không hợp lệ, license thiếu cho external hoặc test chứa source ngoài.
- Báo cáo phân bố theo số box và số ảnh, không chỉ số ảnh.

## Dataset v0 cho lần train Colab đầu tiên

- Mục đích: chứng minh Drive -> manifest -> split -> Colab train -> export -> laptop inference.
- Tối thiểu đề xuất: 5 box khác nhau mỗi lớp, 6-10 ảnh/box; nếu chưa có đủ, ghi rõ là smoke dataset.
- Có train/val; test v0 chỉ là sanity check, không công bố accuracy như kết quả đề tài.
- Có thể chụp webcam/điện thoại rồi lưu lên Drive. Điều kiện không đại diện giàn thật.

## v1 cho kết quả cuối

- Theo mục tiêu khoảng 1.700 ảnh của spec gốc nhưng ưu tiên số box đa dạng hơn số ảnh/box.
- Test box được chọn trước khi chụp và không tái sử dụng trong train/val.
- Ít nhất hai người review 10-15% mỗi lớp; bất đồng quá 5% thì cập nhật label guide và review lại.
- Dataset ngoài phải có `SOURCE`, `LICENSE`, ngày tải và mapping nhãn được duyệt bằng mắt.

## Work items

### L01.1 - Schema và validator

Tạo schema manifest, class enum, checksum, validation report và unit tests.

### L01.2 - ROI/preprocess dataset

Tạo derived square ROI bằng config; lưu config hash. Không áp augmentation vào derived dataset.

### L01.3 - Group split

Implement reserved-test-first, sau đó group-aware train/val. `StratifiedGroupKFold` chỉ là công cụ hỗ trợ; sau split vẫn phải kiểm distribution và có thể cảnh báo nếu một lớp lệch quá tolerance.

### L01.4 - Materialize/report trên Drive

Tạo split archive, `dataset_report.json` và bảng Markdown trong Drive. Output phải tái tạo giống nhau với cùng manifest + seed. Colab giải nén archive vào `/content` cho từng run.

## Acceptance tests

- Cùng box ở hai split -> fail.
- Cùng SHA-256 ở hai split -> fail.
- External ở val/test -> fail.
- Thiếu license external -> fail.
- Seed giống nhau -> output split giống nhau.
- Test session mới + box mới -> pass.
- Mọi class xuất hiện trong train/val; thiếu class -> fail sớm.

## Done

- `dataset_v0.csv`, `dataset_report_v0.json`, split tree và label guide tồn tại.
- Không leakage theo box/hash.
- OpenCode review tập trung vào leakage/provenance đạt APPROVED.
- Commit đề xuất: `feat(data): add versioned group-safe dataset pipeline`.
