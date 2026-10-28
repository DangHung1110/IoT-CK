# OpenCode review - security hardening

Ngày: 2026-10-28

Phạm vi review chỉ gồm `.gitignore`, pre-commit, CI, Dependabot, branch-protection specification và `SECURITY.md`; không gửi spec, dataset hoặc secret.

## Vòng 1

OpenCode yêu cầu làm rõ enforcement của secret scan và branch protection.

## Sửa đổi

- Thêm CI step riêng `pre-commit run detect-secrets --all-files`.
- Đính kèm cấu hình pre-commit để chứng minh hook `detect-secrets` tồn tại.
- Thêm `.github/BRANCH_PROTECTION.md` cho `dev` và `main`.
- Ghi rõ branch ruleset là cấu hình phía GitHub phải bật sau khi push.

## Vòng 2

OpenCode xác nhận:

- Secret scan được cấu hình và enforce trong CI.
- `pip-audit`, Dependabot và `permissions: contents: read` tồn tại.
- Quy tắc bảo vệ `dev`/`main` cụ thể và phân biệt đúng với GitHub-side activation.

Kết quả:

```text
APPROVED
```

Residual P3: approval review tùy quy mô nhóm, chưa dùng secrets baseline và `pip-audit` dùng exit code mặc định làm gate. Không có P0/P1/P2 còn lại.
