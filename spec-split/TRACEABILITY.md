# Ma trận truy vết spec gốc

Mục đích: bảo đảm việc tách spec không làm mất yêu cầu. `NOW` là phạm vi laptop hiện tại; `BACKLOG` được giữ nhưng không triển khai; `REPORT` là nội dung báo cáo/giới hạn.

| Mục spec gốc | Đích mới | Trạng thái |
| --- | --- | --- |
| 0. Tóm tắt/thay đổi | `README`, `00-SPEC-REVIEW` | REPORT |
| 1. Mục tiêu/phạm vi/KPI | `L03`, `L06`, `P00` | NOW + BACKLOG |
| 2. Kiến trúc | `L04`, `L05`, `P00`, `F00` | NOW + BACKLOG |
| 3. Phần cứng | `P00` | BACKLOG |
| 4. Luồng/quyết định | `L03`, `L04` | NOW |
| 5. UART | `P00`, `F00/F01` | BACKLOG |
| 6. QR chống giả | `L04`, `L05` | NOW |
| 7. AI design | `L01`, `L02`, `L03` | NOW |
| 8. Dữ liệu | `L01` | NOW; v1 hoàn tất khi có giàn |
| 9. Train Colab | `L02` | NOW trên Google Colab; không train laptop |
| 10. Eval/export/benchmark | `L03`, `P00/P03` | NOW + BACKLOG |
| 11. Laptop trước/Pi sau | `L00` đến `L06`, `P00` | NOW + BACKLOG |
| 12. Edge Pi | `L04`, `L05`, `P00` | Core NOW; adapter Pi BACKLOG |
| 13. Management Console | `L05` | MVP NOW; RBAC/model/config/OTA BACKLOG |
| 14. ThingsBoard | `P00/P06` | BACKLOG |
| 15. Bảo mật | `L00`, `L04`, `L05`, `P00/P07`, `F00` | Một phần NOW; device BACKLOG |
| 16. Secure OTA | `F00` | BACKLOG ưu tiên cộng điểm |
| 17. Kịch bản demo | `L05`, `L06`, `P00/P08`, `F00` | Laptop NOW; full demo BACKLOG |
| 18. Lộ trình/rủi ro | `README`, mọi phase | ACTIVE |
| Phụ lục A DoD | `README`, `L06`, `P00`, `F00` | ACTIVE |
| Phụ lục B tài liệu | `00-SPEC-REVIEW` và ADR theo phase | ACTIVE |

## Các yêu cầu cố ý hoãn khỏi laptop gate

- Cơ khí, BOM, nguồn, GPIO, camera Pi, motor/servo/sensor.
- UART vật lý và ESP32 firmware.
- ThingsBoard dashboard/RPC/TLS production.
- Pi hardening/systemd/network.
- Console đầy đủ RBAC/2FA/model/config/firmware management.
- Secure OTA/Secure Boot/Flash Encryption.

Các yêu cầu này không bị loại bỏ; chỉ không được phép làm lẫn vào L00-L06.
