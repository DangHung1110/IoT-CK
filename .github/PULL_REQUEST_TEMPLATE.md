## Mục tiêu

Mô tả một mục tiêu duy nhất của PR.

## Spec / work item

Liên kết spec con hoặc issue liên quan.

## Kiểm chứng

- [ ] Quality gate local xanh
- [ ] Test mới bao phủ success, boundary và failure
- [ ] OpenCode review không còn P0/P1
- [ ] Không có secret, dataset hoặc binary lớn
- [ ] Documentation/config đã cập nhật

## ML/data checklist nếu áp dụng

- [ ] Không có split leakage
- [ ] Không dùng test set để tuning
- [ ] Train/export chạy trên Colab, không chạy trên laptop
- [ ] Artifact có model card và SHA-256

## Rủi ro còn lại

Nêu rõ residual risks hoặc ghi `None`.
