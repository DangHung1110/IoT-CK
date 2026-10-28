# Review spec gốc

Ngày review: 2026-10-28. Nguồn: tài liệu spec v1.0 trong thư mục `CK`.

## Kết luận

Spec gốc có kiến trúc hợp lý, tư duy laptop-first, offline-first và fail-safe tốt. Tuy nhiên chưa nên code trực tiếp vì còn một số chỗ mâu thuẫn giữa phần mô tả, code mẫu và tiêu chí kiểm thử. Các mục dưới đây phải được coi là correction bắt buộc cho các spec con.

## Các điểm đúng và nên giữ

- Một code base, thay adapter phần cứng giữa laptop và Raspberry Pi.
- AI chỉ là classification 5 lớp ở phiên bản đầu; detection để sau.
- Split theo `box_id`, không split ngẫu nhiên theo ảnh.
- Validation dùng để chọn model/ngưỡng; test chỉ mở một lần để báo cáo cuối.
- ONNX là hợp đồng triển khai, có model card và kiểm SHA-256.
- QR HMAC, replay protection, UART CRC/whitelist, outbox idempotent và fail-safe đều phù hợp.
- Secure OTA có ký firmware, HTTPS, hash, rollback và chống downgrade là hạng mục cộng điểm tốt.

## Correction bắt buộc trước khi triển khai

| ID | Vấn đề trong spec gốc | Quyết định sửa |
| --- | --- | --- |
| C-01 | Tiền xử lý model chưa được mô tả như một hợp đồng có phiên bản. Code mẫu resize thẳng, trong khi pipeline classification của Ultralytics có transform riêng và thay đổi theo phiên bản. | Tạo `preprocess_id`, ghi đầy đủ ROI, resize/letterbox, RGB, scale, mean/std, layout và output semantics trong model card. Cùng một hàm preprocess được dùng cho eval ONNX và edge. Bắt buộc parity test PyTorch/ONNX. |
| C-02 | `tau_conf`, `tau_damage`, “ngưỡng theo lớp” và đoạn `sweep()` không cùng một logic. Điều này có thể cho KPI đẹp nhưng runtime quyết định khác. | Một hàm policy duy nhất được dùng cho runtime và đánh giá. Dùng `tau_ok_pass` và `tau_damage_by_class`; mọi trường hợp còn lại là `AI_UNCERTAIN` và bị loại. |
| C-03 | Một box có thể xuất hiện ở nhiều lớp; code split mẫu chưa thực thi “phiên test riêng” và stratification theo ảnh có thể lệch. | Gán split ở cấp `box_id`. Box test là box mới, không được từng xuất hiện trong train/val. Script phải kiểm tra leakage, phân bố lớp, source và session rồi mới materialize ảnh. |
| C-04 | “50 ảnh replay đúng 100%” đang trộn integration correctness với model accuracy. | Hai suite tách biệt: fixture với `FakeClassifier` phải đúng 100%; real-model replay chỉ đo metric và không có yêu cầu 100%. |
| C-05 | Vòng lặp mẫu vẫn chạy AI khi QR đã lỗi dù bảng quyết định nói kiểm QR trước để tiết kiệm và hạn chế bề mặt tấn công. | Short-circuit: QR invalid/unreadable/expired/replay/unknown warehouse thì không chạy AI. Có test xác nhận classifier không được gọi. |
| C-06 | `wait_event('DONE')` không kiểm kết quả timeout nhưng vẫn mark seen/log. | Chỉ commit `seen_parcels` sau ACK + DONE thành công; timeout chuyển ERROR. Ghi event/outbox trong transaction có trạng thái rõ ràng. |
| C-07 | Model chính được chốt là YOLOv8n-cls trước khi benchmark. | Chưa gán “model chính”. MobileNetV3-Small và YOLO-cls là hai candidate. Chọn bằng damage recall, false reject, latency và độ ổn định export. |
| C-08 | Dataset v0 khoảng 40 ảnh/lớp quá nhỏ để kết luận chất lượng. | v0 chỉ chứng minh pipeline train-export-infer. Không dùng metric v0 làm kết luận. v1 mới dùng cho KPI. |
| C-09 | Tên lớp ưu tiên chỉ lưu một nhãn, làm mất thông tin ảnh có nhiều hư hại. | Giữ `primary_class` để train single-label, thêm `observed_defects` trong manifest để không mất thông tin và dễ nâng cấp multi-label. |
| C-10 | Python laptop hiện không khớp Python 3.11 của Pi. | Dùng Python 3.11 x64 do `uv` quản lý; không dùng Python 3.12.7 của MSYS2 cho project. |
| C-11 | Dependency không khóa phiên bản; Ultralytics hiện thay đổi model mặc định và transform theo release. | Commit `pyproject.toml` + `uv.lock`; ghi chính xác phiên bản torch, torchvision, ultralytics, onnxruntime và opset trong model card. |
| C-12 | Chống downgrade OTA chỉ dựa semver trong NVS yếu hơn anti-rollback phần cứng. | Bản demo đầu dùng signed manifest + NVS monotonic version + rollback. Hạng mục cộng điểm dùng ESP-IDF Secure Boot v2, signed app verification và security version/eFuse trên board thử riêng. |
| C-13 | Laptop có GPU nhưng người dùng không muốn train local vì tải nhiệt và rủi ro ảnh hưởng máy. | Mọi training, validation tuning và export PyTorch sang ONNX chạy trên Google Colab. Dữ liệu/checkpoint/artifact gốc lưu Google Drive. Laptop chỉ chạy software test và ONNX Runtime inference/benchmark CPU. |

## Những điểm cần điều chỉnh nhưng không chặn giai đoạn laptop

- `accuracy >= 85%` chỉ là KPI mục tiêu; cổng quan trọng hơn là damage recall và false reject trên test độc lập.
- HMAC-SHA256 cắt còn 128 bit vẫn đủ cho bài tập, nhưng cần chặn timestamp quá xa trong tương lai, không chỉ QR quá cũ.
- Giấy phép và số lượng của dataset ngoài phải kiểm lại tại thời điểm tải; không lấy mô tả trong spec làm bằng chứng pháp lý.
- Không đưa model binary, raw dataset, secrets hoặc private key vào Git. Chỉ commit manifest nhỏ, checksum và script tái tạo.
- Secure OTA nên dùng primitive có sẵn của ESP-IDF khi chuyển sang firmware thật; tool desktop chủ yếu làm release/signing UI, không tự phát minh boot chain.

## Nguồn chính thức đã đối chiếu

- Ultralytics classification hiện dùng crop-based transforms mặc định và có `ClassifyLetterBox` khi muốn giữ toàn ảnh: <https://docs.ultralytics.com/tasks/classify>
- ONNX Runtime yêu cầu khóa đúng package/runtime theo target: <https://onnxruntime.ai/docs/install/>
- `StratifiedGroupKFold` giữ group không giao nhau nhưng không đảm bảo split tối ưu trong mọi phân bố: <https://scikit-learn.org/stable/modules/cross_validation.html>
- ESP32 Secure Boot v2 chỉ áp dụng cho ESP32 revision 3.0 trở lên và xác minh app khi OTA: <https://docs.espressif.com/projects/esp-idf/en/stable/esp32/security/secure-boot-v2.html>
- ESP-IDF rollback yêu cầu app mới tự xác nhận sau self-test, nếu không sẽ quay lại app cũ: <https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/ota.html>
