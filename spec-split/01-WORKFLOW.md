# Quy trình làm việc, review và Git

## Nguyên tắc

- Một phase lớn được chia thành work item có thể hoàn thành trong khoảng nửa ngày đến một ngày.
- Mỗi work item chỉ có một mục tiêu, một nhóm test và một commit logic.
- Test phải thất bại vì đúng lý do trước khi code phần logic quan trọng.
- Không push code chưa qua quality gate và OpenCode review.
- Không sửa finding bằng cách xóa/né test hoặc giảm yêu cầu.

## Branch và commit

- Nhánh chính: `main`, luôn chạy được.
- Mỗi work item: `feat/lxx-short-name`, `fix/lxx-short-name`, `test/lxx-short-name`, hoặc `docs/lxx-short-name`.
- Commit theo Conventional Commits, ví dụ `feat(ml): add group-aware dataset split`.
- Không gộp model, QR, console và firmware trong cùng một commit.
- Không force-push `main`. Không commit `.env`, key, certificate private, raw image, `.pt`, `.onnx`, database hoặc output train.

## Vòng lặp cho từng work item

1. Đọc spec con và viết acceptance test/checklist.
2. Tạo branch.
3. Viết test hoặc validator trước.
4. Cài đặt nhỏ nhất để pass.
5. Chạy test gần nhất, rồi chạy toàn bộ quality gate.
6. Chạy OpenCode review ở chế độ chỉ đọc trên diff.
7. Sửa finding, chạy lại test và review nếu diff thay đổi đáng kể.
8. Human review checklist.
9. Commit; push branch; merge khi CI xanh.

## Quality gate chuẩn

```powershell
uv sync --frozen --group dev --group vision-runtime --group qr --group console
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest -q --cov=src --cov-report=term-missing
```

Các notebook không được là nơi duy nhất chứa logic. Logic train/eval/export nằm trong package/script có test; notebook chỉ gọi lại các hàm đó.

Training group không được cài mặc định trên laptop. Colab cài dependency train từ file lock/export riêng, ghi lại runtime và lưu output về Google Drive.

## OpenCode review gate

OpenCode 1.18.35 đã có trên máy nhưng cần provider hoạt động và quyền ghi thư mục session/log của chính OpenCode. Không dùng `--auto` cho review.

Prompt chuẩn:

```text
Bạn là reviewer chỉ đọc. Hãy review diff hiện tại so với main theo spec con được chỉ định.
Không sửa file và không chạy lệnh làm thay đổi repo. Ưu tiên: sai yêu cầu, data leakage,
metric sai, preprocessing train/runtime lệch, security, race/transaction, test thiếu.
Trả về findings theo P0/P1/P2/P3, mỗi finding có file:dòng, tác động và cách sửa.
Nếu không có lỗi chặn, ghi rõ APPROVED và nêu residual risks.
```

Lệnh dự kiến sau khi Git đã tồn tại:

```powershell
opencode run --pure --title "review Lxx" --file ".\specs\Lxx.md" `
  "Review read-only current git diff against the attached spec. Do not edit files. Use the standard severity format."
```

Lưu output review vào `artifacts/reviews/Lxx-YYYYMMDD.md`; thư mục artifact review text được commit, session/cache của OpenCode không commit.

## Human review checklist

- [ ] Diff chỉ đúng phạm vi work item.
- [ ] Test có case thành công, biên và lỗi.
- [ ] Không hard-code class order, secret, path máy cá nhân hoặc timestamp giả.
- [ ] Có seed và provenance cho kết quả ML.
- [ ] Metric runtime và metric eval gọi cùng policy/preprocess.
- [ ] Log không chứa secret hoặc dữ liệu nhạy cảm.
- [ ] Documentation và command chạy lại đã cập nhật.

## Quy tắc push

Chỉ push khi local gate xanh và review đạt. Nếu remote chưa có CI, vẫn phải lưu nguyên output test/review trong mô tả commit hoặc PR. Model artifact được phát hành bằng release/artifact store kèm SHA-256, không push Git thông thường.
