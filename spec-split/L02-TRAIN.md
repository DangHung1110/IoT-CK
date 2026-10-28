# L02 - Train baseline trên Google Colab

## Mục tiêu

Train hai candidate trên cùng dataset/split bằng Google Colab và tạo artifact có thể tái lập. **Không train trên laptop**, kể cả v0. Google Drive là nguồn dữ liệu và nơi lưu checkpoint/model/report.

## Luồng Colab và Drive

1. Colab clone đúng commit của repo.
2. Mount `MyDrive/parcel_sorter`.
3. Cài dependency từ file lock/export dành cho Colab.
4. Kiểm SHA-256 của manifest và split archive.
5. Copy `dataset_vN_splits.zip` từ Drive sang `/content/dataset/` rồi giải nén.
6. Train trên GPU Colab; checkpoint định kỳ và best checkpoint được copy về Drive.
7. Eval/export/report chạy trong cùng môi trường đã khóa phiên bản.
8. Laptop chỉ tải ONNX + model card + checksum để test software.

Không đọc ảnh trực tiếp từ mounted Drive trong vòng train vì sẽ chậm và dễ gián đoạn. Không lưu artifact quan trọng chỉ trong `/content` vì runtime có thể bị reset.

## Candidate

1. MobileNetV3-Small pretrained ImageNet.
2. YOLO classification nano được khóa đúng checkpoint và phiên bản Ultralytics trong lockfile.

Không gọi model nào là “chính” trước L03. YOLOv8n-cls trong spec gốc vẫn là candidate hợp lệ, nhưng không tự động đổi sang model YOLO mới chỉ vì package mới hơn.

## Training contract

Mỗi run lưu trên Drive:

- `run_id`, git commit, dirty flag;
- dataset version + manifest SHA-256 + split seed;
- Colab runtime, Python/package versions, GPU, CUDA runtime;
- model/checkpoint origin và license;
- preprocess/augmentation config đầy đủ;
- seed, epochs, batch, optimizer, scheduler, early-stop rule;
- best checkpoint theo validation objective;
- metrics từng epoch và thời gian train.

## Preprocess contract

- Derived input là ROI vuông; không stretch từ full frame tùy ý.
- Runtime transform được định danh bằng `preprocess_id`.
- Train augmentation nhẹ: flip ngang/dọc khi hợp lý, rotate nhẹ, brightness/contrast nhẹ, scale nhỏ.
- Không dùng random erasing, MixUp/CutMix hoặc crop mạnh ở baseline.
- Validation không augmentation ngẫu nhiên.
- Nếu dùng custom transform cho Ultralytics, validation và export parity phải dùng đúng custom transform đó.

## Work items

### L02.1 - Notebook bootstrap và common experiment runner

Notebook chỉ bootstrap Drive/runtime. Config YAML/Pydantic, seed toàn bộ RNG, run directory, metadata và checkpoint convention nằm trong source code để test được.

### L02.2 - MobileNetV3-Small baseline

- Freeze backbone ngắn, train head.
- Unfreeze và fine-tune với LR thấp.
- Early stop theo validation objective ưu tiên damage recall với false reject constraint, không chỉ accuracy.

### L02.3 - YOLO-cls baseline

- Pin checkpoint + Ultralytics version.
- Dùng cùng split và semantic augmentation gần tương đương candidate 1.
- Xác nhận class order từ artifact; không hard-code.

### L02.4 - Reproducibility smoke trên Colab

Chạy hai run seed giống trên subset nhỏ trong Colab; class mapping, split và metric pipeline phải giống. GPU có thể không deterministic tuyệt đối, nên đặt tolerance và ghi rõ.

## Quản lý runtime Colab

- Chọn GPU runtime khi bắt đầu session và ghi model GPU vào metadata.
- Bắt đầu batch theo VRAM thực tế của runtime; nếu out-of-memory, giảm batch trước, không giảm ROI/nhãn âm thầm.
- Dùng AMP nếu metric/parity ổn.
- Lưu checkpoint về Drive định kỳ để resume sau khi Colab ngắt phiên.
- Không dùng test set để chọn batch, epoch, model hoặc threshold.
- Laptop không cài training stack và không chạy notebook train.

## Acceptance tests

- Overfit được một mini-batch nhỏ: chứng minh loss/model wiring đúng.
- Class index trong trainer khớp manifest.
- Resume checkpoint không đổi run identity/dataset identity.
- Validation loader không có random transform.
- Mỗi run tạo metadata đủ trường và checkpoint có checksum.

## Done

- Có ít nhất một run Colab hoàn chỉnh cho mỗi candidate trên v0.
- Command chạy lại được từ repo sạch với dataset cùng checksum.
- Checkpoint, metadata, report và export được lưu trên Google Drive.
- Chưa tuyên bố đạt KPI từ v0.
- OpenCode review tập trung preprocessing, split usage và reproducibility đạt APPROVED.
- Commit đề xuất: `feat(ml): add reproducible classification baselines`.
