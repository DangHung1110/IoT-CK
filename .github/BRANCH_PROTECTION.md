# GitHub branch protection

Repository setup chưa hoàn tất cho đến khi các rule dưới đây được bật trong GitHub Settings.

## `dev`

- Chỉ nhận thay đổi qua pull request từ `feat/*`, `fix/*`, `chore/*`, `docs/*`, hoặc `test/*`.
- Require status check `Environment / validate` trước merge.
- Require conversation resolution.
- Block force pushes và branch deletion.
- Không cho phép bypass status check.
- Nếu nhóm có từ hai thành viên trở lên: require ít nhất một approval và dismiss stale approvals.

## `main`

- Chỉ merge từ `dev` khi đạt release gate.
- Require pull request và status check `Environment / validate`.
- Require conversation resolution và linear history.
- Block force pushes và branch deletion.
- Không cho phép bypass rule trong thao tác thông thường.
- Không commit hoặc push trực tiếp vào `main`.

## Kiểm tra sau cấu hình

1. Thử push trực tiếp một commit thử nghiệm không chứa thay đổi vào `main`/`dev`: GitHub phải từ chối.
2. Tạo PR từ nhánh công việc vào `dev`: chỉ merge được sau khi `Environment / validate` xanh.
3. Xác nhận nhánh `main` vẫn chỉ chứa bản phát hành cuối.

Branch protection là cấu hình phía GitHub, không nằm trong Git history. Chủ repository phải kiểm tra lại ruleset sau khi chuyển quyền sở hữu, đổi gói GitHub hoặc thay đổi workflow/check name.
