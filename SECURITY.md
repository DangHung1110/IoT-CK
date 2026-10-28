# Security policy

## Supported branches

- `main`: bản phát hành cuối đã nghiệm thu.
- `dev`: bản tích hợp đang được kiểm thử.
- Nhánh công việc không được coi là bản phát hành hỗ trợ.

## Không được đưa vào Git

- `.env` thật, API token, mật khẩu, cookie hoặc access token.
- Khóa QR HMAC, private key OTA, private key CA, khóa Secure Boot.
- Device certificate/private key, SSH private key hoặc Google service-account credential.
- Dataset ảnh thật, ảnh người dùng, database local, log runtime.
- Checkpoint, ONNX, firmware binary hoặc artifact build lớn.

Các file này phải nằm trong Google Drive có kiểm soát truy cập, secret store, hoặc trực tiếp trên thiết bị với quyền file phù hợp. Git chỉ lưu schema, tài liệu, checksum và file `.env.example` không chứa giá trị thật.

## Kiểm tra bắt buộc trước push

```powershell
uv sync --frozen
uv run pre-commit run --all-files
uv run pip-audit
git diff --cached --check
```

Mọi PR vào `dev` phải qua secret scan, dependency audit và review. Không dùng `--no-verify` để bỏ qua hook.

Local hook không phải ranh giới bảo mật duy nhất. CI chạy riêng hook `detect-secrets`, còn GitHub phải bật ruleset theo [`.github/BRANCH_PROTECTION.md`](.github/BRANCH_PROTECTION.md) để chặn direct push, force-push và merge khi status check chưa đạt.

## Khi phát hiện secret đã commit

1. Dừng push/merge ngay lập tức.
2. Thu hồi hoặc rotate secret trước; xóa file không làm secret cũ hết hiệu lực.
3. Thông báo cho người phụ trách repository.
4. Làm sạch lịch sử bằng quy trình được duyệt riêng và force-push chỉ khi đã phối hợp với toàn nhóm.
5. Kiểm tra log/audit để đánh giá secret có bị sử dụng trái phép hay không.

Không đăng secret thật vào issue, pull request, ảnh chụp màn hình hoặc báo cáo review.

## Báo cáo lỗ hổng

Không mở public issue chứa exploit hoặc credential. Gửi báo cáo riêng cho chủ repository, kèm phạm vi ảnh hưởng, cách tái hiện an toàn và phiên bản liên quan.
