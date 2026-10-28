# Parcel Sorter - bộ spec đã tách

Bộ tài liệu này thay thế cách triển khai trực tiếp từ spec dài. Spec gốc vẫn là tài liệu yêu cầu tổng thể; các file ở đây là đơn vị thực thi, kiểm thử, review và commit.

## Phạm vi hiện tại

Chỉ làm trên laptop:

1. môi trường và repo;
2. dữ liệu v0/v1 contract, dữ liệu gốc lưu trên Google Drive;
3. train hai model candidate trên Google Colab, không train trên laptop;
4. đánh giá và export ONNX trên Colab, chỉ benchmark ONNX trên laptop;
5. QR, policy quyết định, edge core với mock/folder camera;
6. lưu trữ, uplink và console tối thiểu;
7. integration gate trên laptop.

Chưa làm trong vòng hiện tại: Raspberry Pi, camera Pi, UART thật, ESP32, băng chuyền, servo, cảm biến, ThingsBoard production, Secure Boot/eFuse. Các phần đó được giữ trong backlog có điều kiện đầu vào rõ ràng.

## Thứ tự bắt buộc

| Thứ tự | Spec | Kết quả |
| --- | --- | --- |
| 0 | `00-SPEC-REVIEW.md` | Chấp nhận correction |
| 1 | `01-WORKFLOW.md` | Có quy trình branch/review/commit |
| 2 | `L00-BOOTSTRAP.md` | Repo và CI xanh |
| 3 | `L01-DATA.md` | Google Drive dataset contract + split không leakage |
| 4 | `L02-TRAIN.md` | Hai baseline train trên Google Colab |
| 5 | `L03-EVAL-EXPORT.md` | Eval/export trên Colab, ONNX benchmark laptop |
| 6 | `L04-CORE-SOFTWARE.md` | QR + decision + edge mock có unit test |
| 7 | `L05-INTEGRATION.md` | Replay, storage, outbox, console MVP |
| 8 | `L06-LAPTOP-GATE.md` | Toàn bộ laptop gate đạt |
| 9 | `P00-PI-BACKLOG.md` | Kế hoạch chuyển Pi, chưa thực thi |
| 10 | `F00-FIRMWARE-SECURITY-BACKLOG.md` | Kế hoạch firmware/OTA cộng điểm, chưa thực thi |

`TRACEABILITY.md` ánh xạ toàn bộ mục 0-18 và phụ lục của spec gốc, để bảo đảm không mất yêu cầu khi tách phase.

Không bắt đầu spec tiếp theo nếu spec hiện tại chưa qua test, OpenCode review và human review.

## Definition of Ready chung

- Input và dependency của phase đã tồn tại.
- Không còn câu hỏi làm thay đổi kiến trúc của phase.
- Không chứa secret, raw dataset hay binary lớn trong Git.
- Test command và tiêu chí pass/fail đã được viết trước khi code.

## Definition of Done chung

- Acceptance tests của phase xanh.
- `ruff check`, `ruff format --check`, `mypy` (vùng production) và `pytest` xanh.
- Coverage cho module logic mới đạt ít nhất 90%; không dùng coverage để thay thế test case chất lượng.
- OpenCode review không còn finding P0/P1; P2 phải sửa hoặc ghi quyết định có lý do.
- Diff được người làm đọc lại, không có secret/binary/cache.
- Chỉ sau đó mới commit và push.

## Trạng thái

| Phase | Trạng thái |
| --- | --- |
| Review và chia spec cục bộ | DONE |
| OpenCode review bộ spec | PENDING - cần xác nhận cho phép gửi tài liệu tới provider OpenCode đang cấu hình |
| L00-L06 | NOT STARTED |
| Pi/hardware/firmware | BACKLOG |
