# Spec đề tài: Hệ thống phân loại kiện hàng thông minh

**QR + AI kiểm tra hư hại bao bì + bảo mật** · Phiên bản 1.0 · 28/10/2026 · Build từ đầu: test trên laptop trước, triển khai lên Raspberry Pi 4 sau, train trên Google Colab, data trên Google Drive.

## 0. Tóm tắt và thay đổi so với đề cũ

Hệ thống nhận một kiện hàng trên băng chuyền, đọc mã QR để biết kiện đi kho nào, dùng camera + AI để kiểm tra hộp có hư hại không, rồi gạt kiện sang khay A, khay B hoặc khay Loại. Mọi sự kiện được ghi log, đẩy lên dashboard ThingsBoard và lưu vào phần mềm quản lý riêng (Management Console).

| Hạng mục | Đề cũ | Spec này |
| --- | --- | --- |
| AI | 2 lớp: OK / Lỗi | 5 lớp: OK, Rách, Móp, Vỡ/Bể, Ướt-bẩn, kèm ngưỡng "không chắc chắn" |
| Model | YOLOv8n-cls hoặc MobileNetV2 | YOLOv8n-cls (khuyến nghị) và MobileNetV3-Small (đối chứng), xuất ONNX, chạy bằng ONNX Runtime |
| Môi trường | Làm thẳng trên Pi | 2 giai đoạn: laptop (giả lập phần cứng) rồi mới đẩy sang Pi 4 |
| Dữ liệu | Chụp trên băng chuyền | Chụp trên băng chuyền là chính, dataset công khai bổ trợ, lưu trên Google Drive, có vòng phản hồi từ Console |
| Phần mềm | Chỉ ThingsBoard | ThingsBoard (realtime) + Management Console (đăng ký kiện, in QR, duyệt ảnh lỗi, quản lý model/cấu hình, báo cáo) |
| Bảo mật | Có | Giữ nguyên, bổ sung cách làm và cách demo từng điểm |

Về yêu cầu "rách, hư, vỡ, bể": mình map thành các lớp cụ thể để AI học được. "Hư" là từ chung nên không làm lớp riêng, mọi lớp khác OK đều là hộp hư và sẽ bị loại.

## 1. Mục tiêu, phạm vi, tiêu chí thành công

### 1.1 Phạm vi theo mức ưu tiên

| Mức | Nội dung |
| --- | --- |
| Must (bắt buộc) | Mô hình băng chuyền chạy thật; đọc QR có HMAC; AI 5 lớp; gạt 3 hướng (A, B, Loại); MQTT qua TLS lên ThingsBoard; dashboard hiển thị số lượng, log, ảnh hàng lỗi; UART Pi-ESP32 có CRC; Console tối thiểu (đăng ký kiện, in QR, xem sự kiện) |
| Should (nên có) | App nạp firmware bảo mật (Secure OTA); duyệt và sửa nhãn ảnh lỗi để retrain; quản lý model theo phiên bản; đăng nhập 2FA |
| Could (để sau) | Detection khoanh vùng vết hỏng; camera thứ 2 xem mặt bên; lượng tử hóa INT8; Secure Boot + Flash Encryption trên ESP32 |

### 1.2 Giới hạn đã biết (nói thẳng trong báo cáo)

- Một camera trên đỉnh chỉ thấy mặt trên của hộp. Hư hại ở mặt dưới hoặc mặt bên khuất sẽ không phát hiện được. Cách khắc phục nếu cần: thêm gương nghiêng hoặc webcam USB thứ 2.
- Chỉ xử lý hộp carton, chụp theo kiểu dừng rồi chụp (không chụp khi băng chuyền đang chạy).
- Vết rách nhỏ và móp nhẹ là loại khó nhất. AI sẽ có ngưỡng không chắc chắn, hộp ở vùng này bị đẩy sang khay Loại để người kiểm tra (fail-safe).
- Dataset công khai trên mạng chủ yếu là nhị phân (hỏng / không hỏng) và khác môi trường chụp của bạn, chỉ dùng bổ trợ.

### 1.3 Tiêu chí chấp nhận (KPI)

Đây là mục tiêu để đo và báo cáo, không phải con số đảm bảo trước. Đo trên tập test gồm các hộp chưa từng xuất hiện trong train/val.

| Chỉ số | Mục tiêu |
| --- | --- |
| Accuracy 5 lớp (tập test) | ≥ 85%, cố gắng 90% |
| Recall phát hiện hộp hư (gộp 4 lớp lỗi so với OK) | ≥ 92% |
| Tỉ lệ loại nhầm hộp tốt (false reject) | ≤ 10% |
| Đọc QR đúng khi QR in sạch | ≥ 98% trong tối đa 3 khung hình |
| Thời gian suy luận AI trên Pi 4 (p95, mỗi khung) | ≤ 250 ms |
| Thời gian từ lúc dừng băng chuyền đến lúc có quyết định | ≤ 1.0 s |
| Chạy liên tục | 100 kiện / 30 phút không treo |
| QR giả, QR trùng (replay) bị chặn | 100% trong kịch bản test |

## 2. Kiến trúc tổng thể

```
  MAY DEV / SERVER (cung mang LAN)
  +---------------+  +--------------------+  +--------------------+
  | ThingsBoard   |  | Management Console |  | OTA Tool (GUI)     |
  | MQTT TLS 8883 |  | FastAPI + DB       |  | giu private key    |
  +-------^-------+  +---------^----------+  +----------+---------+
          | MQTT/TLS           | HTTPS                  | .bin + manifest
          |                    |                        v
  +-------+--------------------+---------+     HTTPS server (Console)
  | RASPBERRY PI 4                       |            ^
  | Camera | QR | AI (ONNX) | Decision   |            | ESP32 tai firmware
  | MQTT | queue offline (SQLite)        |  UART 115200 + CRC16
  +-------^------------------------------+<----------------------+
          | CSI                           |   +-------------------+ |
   Pi Camera + den LED co dinh            +-->| ESP32             |-+
                                              | Conveyor, Servo,  |
                                              | Sensor, LED, Buzzer|
                                              +-------------------+
```

### 2.1 Các khối và công nghệ

| Khối | Vai trò | Công nghệ |
| --- | --- | --- |
| Pi edge app | Chụp ảnh, đọc QR, chạy AI, ra quyết định, gửi lệnh UART, gửi MQTT, hàng đợi offline | Python 3.11, picamera2, pyzbar, onnxruntime, paho-mqtt, SQLite |
| ESP32 firmware | Điều khiển băng chuyền, cảm biến, servo, LED, còi; chạy OTA | PlatformIO. Arduino framework là đủ; nếu muốn Secure Boot thì dùng framework espidf |
| ThingsBoard | Dashboard realtime, nút điều khiển (RPC), cảnh báo | ThingsBoard CE chạy bằng Docker |
| Management Console | Đăng ký kiện, in QR, duyệt ảnh lỗi, thống kê, quản lý model và cấu hình | FastAPI, SQLite (có thể đổi PostgreSQL), Jinja2 + HTMX |
| Training pipeline | Chuẩn bị data, train, đánh giá, xuất ONNX | Google Colab + Google Drive |
| OTA Tool | Ký và phát hành firmware | Python GUI (Tkinter hoặc PySide6), thư viện cryptography |

### 2.2 Nguyên tắc thiết kế

1. **Pi quyết định, ESP32 chỉ thực thi.** ESP32 không tự hiểu QR hay AI, chỉ nhận lệnh trong whitelist.
2. **Offline-first.** Pi phải tự quyết định và chạy được khi mất mạng: kiểm HMAC cục bộ, log vào SQLite, đồng bộ lên ThingsBoard và Console khi có mạng lại (store-and-forward).
3. **Fail-safe.** Bất cứ chỗ nào nghi ngờ (QR lỗi, AI không chắc, mất liên lạc) thì loại hộp hoặc dừng băng chuyền, không bao giờ cho qua.
4. **Không tin dữ liệu bên ngoài.** QR, lệnh RPC, file firmware đều phải xác thực trước khi dùng.
5. **Cùng một code chạy ở laptop và Pi.** Chỉ thay lớp phần cứng (camera, serial) bằng cấu hình, xem mục 11.

## 3. Phần cứng chi tiết

### 3.1 Danh sách linh kiện (BOM)

| Linh kiện | SL | Ghi chú |
| --- | --- | --- |
| Raspberry Pi 4 (khuyến nghị 4GB) | 1 | Cài Raspberry Pi OS 64-bit (cần cho onnxruntime). Nguồn 5V/3A chính hãng, thẻ nhớ 32GB A1/A2, tản nhiệt hoặc quạt vì suy luận liên tục dễ làm Pi nóng và giảm xung |
| Pi Camera Module 3 (autofocus) hoặc webcam USB tốt | 1 | Với Module 3 nên khóa lấy nét thủ công (LensPosition) để ảnh ổn định. Webcam USB giúp test trên laptop bằng cùng thiết bị |
| ESP32 DevKit (WROOM-32) | 1 | Không dùng bản WROVER vì GPIO16/17 bị PSRAM chiếm |
| Motor DC giảm tốc 12V + mạch L298N | 1 | Băng chuyền chạy chậm, mô-men đủ kéo hộp thử nghiệm |
| Servo MG996R (hộp nặng) hoặc SG90 (hộp nhẹ) | 2 | Cấp nguồn riêng 5-6V, tối thiểu 2A cho 2 servo |
| TCRT5000 (khoảng cách 1-2 cm) hoặc HC-SR04 | 1 (tùy chọn 2) | TCRT5000 đơn giản hơn. Cảm biến thứ 2 tại vị trí gạt giúp gạt đúng lúc |
| LED xanh, LED đỏ, điện trở 220 ohm, còi buzzer active | 1 bộ |  |
| Dải LED trắng 12V (5000-6500K) + tấm khuếch tán | 1 | Ánh sáng cố định, tránh bóng đổ và lóa. Nguồn riêng |
| Nguồn 12V 3A, mạch hạ áp 12V xuống 5V 3A | 1 | Hạ áp cấp cho servo |
| Nút dừng khẩn cấp vật lý | 1 | Cắt nguồn motor độc lập với phần mềm |
| Khung camera, hộp che sáng ngoài | 1 | Che ánh sáng phòng để ảnh giống nhau mọi lúc |
| Hộp carton thử nghiệm | ≥ 150 | Xem mục 8 về dữ liệu |

### 3.2 Nguyên tắc cấp nguồn và nối đất

- Nối chung GND của Pi, ESP32, L298N, nguồn servo.
- Không lấy 5V từ chân ESP32 hoặc Pi để cấp cho servo hay motor. Servo giật dòng sẽ làm ESP32 và Pi reset.
- Thêm tụ 470-1000 uF gần nguồn servo.
- Pi và ESP32 đều là mức logic 3.3V nên nối UART trực tiếp, nối chéo TX với RX.
- Chân ECHO của HC-SR04 xuất 5V, phải qua cầu chia áp (1k + 2k ohm) trước khi vào ESP32.

### 3.3 Sơ đồ chân đề xuất

| Chức năng | ESP32 | Ghi chú |
| --- | --- | --- |
| UART2 TX gửi sang Pi | GPIO17 | Nối vào Pi RXD (GPIO15, chân vật lý 10) |
| UART2 RX nhận từ Pi | GPIO16 | Nối vào Pi TXD (GPIO14, chân vật lý 8) |
| Motor ENA (PWM) | GPIO25 |  |
| Motor IN1 / IN2 | GPIO26 / GPIO27 |  |
| Servo A / Servo B | GPIO18 / GPIO19 | Dùng thư viện ESP32Servo |
| Cảm biến TCRT5000 (DO) | GPIO34 | Chân chỉ-vào, phù hợp |
| HC-SR04 TRIG / ECHO | GPIO32 / GPIO33 | ECHO qua chia áp |
| LED xanh / LED đỏ / Buzzer | GPIO21 / GPIO22 / GPIO23 |  |
| Nút E-Stop (đọc trạng thái) | GPIO4 | INPUT\_PULLUP |

Tránh các chân strapping GPIO0, 2, 5, 12, 15 khi chưa hiểu rõ ảnh hưởng đến lúc khởi động.

### 3.4 Cơ khí và bố trí

- Camera nhìn thẳng xuống, cao khoảng 25-35 cm tùy góc nhìn, vùng chụp cố định phủ trọn hộp lớn nhất. Hộp nên chiếm 50-80% khung hình. Cắt ảnh (ROI) cố định ở phần mềm.
- Cảm biến đặt trước vùng chụp một đoạn bằng quãng phanh của băng chuyền. Hiệu chỉnh bằng thực nghiệm thông số STOP\_DELAY\_MS (từ lúc cảm biến thấy hộp đến lúc dừng motor).
- Hộp che sáng bao quanh vùng chụp, đèn LED đặt hai bên hoặc dạng vòng, có tấm khuếch tán.
- Mặt băng chuyền nên đơn sắc, tương phản với carton (đen mờ hoặc xanh lá). Điều kiện này phải giữ nguyên khi chụp dataset và khi chạy thật.
- Khay A và B ở hai bên, khay Loại ở cuối băng. Servo có hai góc: nghỉ và gạt, hiệu chỉnh trên thực tế (ví dụ 0 độ và 70 độ).
- Thời điểm gạt tính bằng T\_TRAVEL\_MS (thời gian hộp đi từ vùng chụp đến servo), tinh chỉnh bằng thực nghiệm.

## 4. Luồng xử lý và quy tắc quyết định

### 4.1 Máy trạng thái trên Pi

| Trạng thái | Điều kiện vào | Hành động | Chuyển sang |
| --- | --- | --- | --- |
| IDLE | Khởi động xong, ESP32 trả PONG | Chờ sự kiện DETECT | DETECTED |
| DETECTED | Nhận EVT DETECT | Chờ ESP32 dừng băng chuyền | SETTLING (khi nhận EVT STOPPED), ERROR (quá 2 s không dừng) |
| SETTLING | Nhận EVT STOPPED | Chờ SETTLE\_MS (300-500 ms) cho hộp hết rung, đèn ổn định | CAPTURE |
| CAPTURE | Hết SETTLE\_MS | Chụp N=3 khung cách nhau khoảng 80 ms | ANALYZE |
| ANALYZE | Có ảnh | Giải mã QR, kiểm HMAC; chạy AI trên 3 khung, lấy trung bình xác suất | DECIDE |
| DECIDE | Có kết quả QR + AI | Áp dụng bảng quyết định mục 4.2 | ACTUATE |
| ACTUATE | Có quyết định | Gửi lệnh SORT, chờ ACK và EVT DONE (timeout 5 s) | LOGGING, ERROR |
| LOGGING | Nhận DONE | Ghi SQLite, gửi MQTT và Console (chạy nền, không chặn) | IDLE |
| ERROR / E-STOP | Mất liên lạc, kẹt hộp, bấm E-Stop | Dừng băng chuyền, đèn đỏ, báo dashboard | IDLE sau khi người vận hành CLEAR |

### 4.2 Bảng quyết định (xét từ trên xuống, khớp dòng nào dừng ở dòng đó)

| Ưu tiên | Điều kiện | Kết quả | Mã lý do |
| --- | --- | --- | --- |
| 1 | Không đọc được QR sau 3 khung | Khay Loại, đèn đỏ, còi | QR\_UNREADABLE |
| 2 | QR sai định dạng hoặc sai HMAC | Khay Loại, cảnh báo bảo mật | QR\_INVALID |
| 3 | QR hết hạn (ngoài cửa sổ thời gian cho phép) | Khay Loại | QR\_EXPIRED |
| 4 | Mã kiện đã được quét trước đó | Khay Loại, cảnh báo bảo mật | QR\_REPLAY |
| 5 | Kho không thuộc A hoặc B | Khay Loại | WAREHOUSE\_UNKNOWN |
| 6 | AI lỗi chạy hoặc quá thời gian | Khay Loại | AI\_ERROR |
| 7 | AI dự đoán lớp lỗi (rach, mop, vo, uot) với xác suất ≥ ngưỡng lớp | Khay Loại, lưu ảnh | DAMAGE\_RACH, DAMAGE\_MOP, DAMAGE\_VO, DAMAGE\_UOT |
| 8 | Xác suất cao nhất < ngưỡng tin cậy | Khay Loại, đánh dấu cần kiểm tra thủ công | AI\_UNCERTAIN |
| 9 | AI = ok và kho A | Khay A | OK\_A |
| 10 | AI = ok và kho B | Khay B | OK\_B |

Hai điểm cần chú ý:

- Mã kiện bị loại vì hư hại sẽ bị coi là đã dùng. Nếu người vận hành đóng gói lại và gửi lại, họ dùng chức năng "Cho phép quét lại" trong Console. Pi nhận danh sách mở khóa khi đồng bộ cấu hình.
- Kiểm tra bảo mật (dòng 1-5) chạy trước AI để tiết kiệm thời gian và tránh lộ thông tin AI cho kẻ tấn công.

### 4.3 Chính sách theo lớp (cấu hình được, đồng bộ từ Console)

| Lớp AI | Hành động mặc định | Ghi chú |
| --- | --- | --- |
| ok | PASS | Đi theo kho trong QR |
| rach | REJECT | Rách bìa, bung băng keo, xé mép |
| mop | REJECT | Móp, méo, bẹp góc. Có thể đổi thành PASS\_FLAG (cho qua nhưng gắn cờ) nếu móp nhẹ chấp nhận được |
| vo | REJECT | Thủng, nát, sập góc, lộ hàng |
| uot | REJECT | Ướt, ố, bẩn nặng |

Ngưỡng khởi đầu: ngưỡng lớp lỗi 0.50, ngưỡng tin cậy 0.60. Chỉnh lại dựa trên tập validation để đạt KPI (xem mục 10), lưu trong file cấu hình.

## 5. Giao thức UART Pi và ESP32

### 5.1 Khung dữ liệu

- Cấu hình: 115200 baud, 8N1.
- Khung ASCII một dòng: `CMD,SEQ,ARG*CRC16\n`
- CRC16-CCITT (đa thức 0x1021, giá trị đầu 0xFFFF) tính trên chuỗi từ CMD đến hết ARG, viết thành 4 ký tự hex in hoa. Test vector: chuỗi `123456789` cho kết quả `29B1`.
- Ví dụ: `SORT,42,A*XXXX\n` (XXXX là CRC).

```python
def crc16_ccitt(data: bytes, crc=0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc

def make_frame(cmd, seq, arg=''):
    body = f'{cmd},{seq},{arg}'.encode()
    return body + f'*{crc16_ccitt(body):04X}\n'.encode()
```

### 5.2 Danh sách lệnh (whitelist)

| Hướng | Lệnh | Ý nghĩa |
| --- | --- | --- |
| Pi sang ESP32 | PING | Kiểm tra sống, gửi mỗi 1 s, ESP32 trả PONG |
| Pi sang ESP32 | START / STOP | Chạy hoặc dừng băng chuyền |
| Pi sang ESP32 | SPEED,n | Tốc độ 0-100 (%), giới hạn trần ở ESP32 |
| Pi sang ESP32 | SORT,A / SORT,B / SORT,R | Chạy băng và gạt về khay A, B hoặc Loại. Chỉ nhận khi đang STOPPED |
| Pi sang ESP32 | CLEAR | Xóa trạng thái lỗi sau E-Stop hoặc kẹt |
| Pi sang ESP32 | OTA,manifest\_url | Chỉ nhận khi Admin kích hoạt, xem mục 16 |
| ESP32 sang Pi | ACK,seq / NAK,seq,code | Xác nhận hoặc từ chối (code: CRC, CMD, STATE, SEQ) |
| ESP32 sang Pi | EVT,seq,DETECT / STOPPED / DONE / JAM / ESTOP | Sự kiện từ cảm biến và cơ cấu |
| ESP32 sang Pi | EVT,seq,OTA\_OK / OTA\_FAIL | Kết quả cập nhật firmware |

### 5.3 Quy tắc độ tin cậy

- Mỗi lệnh Pi gửi phải được ESP32 ACK trong 200 ms. Quá hạn thì gửi lại tối đa 3 lần, sau đó báo lỗi.
- CRC sai, lệnh ngoài whitelist, tham số ngoài khoảng: trả NAK và bỏ qua.
- SEQ tăng dần. ESP32 từ chối SEQ lùi (chống phát lại lệnh trên dây), trừ khi vừa reset.
- Watchdog: ESP32 không nhận khung hợp lệ nào trong 3 s thì dừng băng chuyền, nháy đèn đỏ.
- Nút E-Stop vật lý cắt nguồn motor, không phụ thuộc phần mềm.

## 6. Mã QR chống giả

### 6.1 Định dạng

`PSv1|<parcel_id>|<kho>|<ts_utc>|<mac>`

Ví dụ: `PSv1|PKG000123|A|20261028T101500Z|<32 ký tự hex>`

- `parcel_id`: chữ hoa và số, 6-20 ký tự.
- `kho`: A hoặc B.
- `ts_utc`: thời điểm tạo, dạng YYYYMMDDTHHMMSSZ.
- `mac`: 32 ký tự hex đầu của HMAC-SHA256(khóa bí mật, chuỗi `PSv1|parcel_id|kho|ts_utc`).

### 6.2 Thuật toán kiểm tra trên Pi

```python
import hmac, hashlib, re

def sign(key: bytes, pid, kho, ts) -> str:
    msg = f'PSv1|{pid}|{kho}|{ts}'.encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()[:32]

def verify(key, qr_text, now_utc, max_age_hours, seen_db):
    p = qr_text.strip().split('|')
    if len(p) != 5 or p[0] != 'PSv1':
        return 'QR_INVALID', None
    _, pid, kho, ts, mac = p
    if not re.fullmatch(r'[A-Z0-9]{6,20}', pid) or not re.fullmatch(r'\d{8}T\d{6}Z', ts):
        return 'QR_INVALID', None
    if not hmac.compare_digest(mac, sign(key, pid, kho, ts)):
        return 'QR_INVALID', None
    # kiem tra han: so sanh ts voi now_utc, vuot max_age_hours -> 'QR_EXPIRED'
    # kiem tra trung: pid da co trong seen_db -> 'QR_REPLAY'
    # kho khong thuoc A,B -> 'WAREHOUSE_UNKNOWN'
    return 'OK', {'parcel_id': pid, 'warehouse': kho, 'ts': ts}
```

- Luôn dùng `hmac.compare_digest` để so sánh, không dùng dấu `==`.
- Khóa: 32 byte ngẫu nhiên tạo bằng `secrets.token_hex(32)`. Trên Pi lưu ở `/etc/sorter/hmac.key`, quyền 600, chủ sở hữu là user chạy dịch vụ. Trên Console lưu bằng biến môi trường, không đưa vào git.
- Chống phát lại: bảng SQLite cục bộ `seen_parcels(parcel_id PRIMARY KEY, first_seen, result)`. Mã đã có thì trả QR\_REPLAY.
- Kiểm tra hạn cần đồng hồ đúng: Pi bật NTP. Nếu không có mạng thường xuyên thì thêm module RTC DS3231, hoặc tắt kiểm tra hạn và ghi chú rõ trong báo cáo.

### 6.3 In và đọc QR

- Cạnh QR tối thiểu 3 x 3 cm, viền trắng (quiet zone) 4 module, mức sửa lỗi M, in đen trên nền trắng mờ (tránh giấy bóng gây lóa). Dán lên mặt trên của hộp, xoay hướng nào cũng được.
- Giải mã: dùng pyzbar làm chính, OpenCV QRCodeDetector làm dự phòng. Tiền xử lý: chuyển xám, tăng tương phản CLAHE, nếu thất bại thì thử xoay 90 độ. Thử lần lượt trên 3 khung đã chụp.
- Console tạo QR và file PDF nhãn in (xem mục 13).

## 7. AI phân loại hư hại: thiết kế

### 7.1 Bài toán và định nghĩa các lớp

Bài toán: phân loại ảnh (image classification) cho một hộp đã dừng yên trong vùng chụp, 5 lớp. Chọn classification thay vì object detection vì nhẹ, chạy nhanh trên Pi 4, gán nhãn đơn giản (nhãn là tên thư mục) và đủ trả lời câu hỏi "kiện này có lỗi không, lỗi loại gì". Đổi lại, model không cho biết vị trí vết hỏng. Nếu cần hiển thị vùng nghi ngờ trên Console, dùng Grad-CAM ở phía laptop, hoặc nâng cấp lên detection ở giai đoạn Could.

| Lớp (tên thư mục) | Định nghĩa | Ví dụ | Không nhầm với |
| --- | --- | --- | --- |
| `ok` | Hộp nguyên vẹn | Nếp nhăn nhẹ, băng keo nhiều lớp, nhãn dán, chữ in, bụi nhẹ, bóng đổ | Đây là hard negative, phải có đủ để model không báo lỗi bừa |
| `rach` | Bìa carton hoặc băng keo bị rách, xé, bung | Mép nắp bị xé, vết rạch dài, băng keo bung, nắp hé | Chưa thủng xuyên đến mức lộ hàng (đó là `vo`) |
| `mop` | Biến dạng hình học nhưng bìa chưa rách | Móp góc, bẹp một mặt, méo cạnh, lõm | Nếp nhăn nhẹ vẫn là `ok` |
| `vo` | Vỡ, bể, thủng, nát | Lỗ thủng, nát góc, sập nắp, lộ hàng bên trong | Rách nhỏ (đó là `rach`) |
| `uot` | Ướt, ố, bẩn nặng | Vệt nước làm sẫm màu, bìa nhão, mốc, bùn dầu | Bụi nhẹ vẫn là `ok` |

Quy tắc khi một ảnh có nhiều lỗi: gán lớp nặng nhất theo thứ tự `vo` > `rach` > `mop` > `uot` > `ok`. Cách này giữ bài toán ở dạng một nhãn, dễ train. Nếu sau này cần biết đủ mọi lỗi cùng lúc thì chuyển sang multi-label (hàm sigmoid), đó là việc của giai đoạn sau.

### 7.2 Chọn model nhẹ cho Pi 4

| Model | Tham số (xấp xỉ) | ONNX FP32 | Ghi chú |
| --- | --- | --- | --- |
| YOLOv8n-cls | 2.7 triệu | khoảng 11 MB | Train và export một dòng lệnh bằng Ultralytics, dễ nhất khi làm trên Colab. Giấy phép AGPL-3.0: dùng học tập thì ổn, thương mại cần xem lại |
| MobileNetV3-Small | 2.5 triệu | khoảng 10 MB | Nhẹ và nhanh nhất, trọng số torchvision/timm giấy phép dễ chịu. Cần tự viết vòng train (có mẫu ở 9.3) |
| MobileNetV2 | 3.5 triệu | khoảng 14 MB | Đã nằm trong đề cũ, dùng làm mốc so sánh nếu cần |
| EfficientNet-Lite0 | 4.7 triệu | khoảng 18 MB | Chính xác hơn một chút nhưng nặng hơn, chỉ thử nếu hai model trên thiếu độ chính xác |

Quyết định: train cả **YOLOv8n-cls (chính)** và **MobileNetV3-Small (đối chứng)**, cùng dataset, cùng kích thước ảnh. Chọn model dựa trên ba tiêu chí theo thứ tự: (1) recall phát hiện hư hại đạt KPI, (2) độ trễ p95 trên Pi 4 ≤ 250 ms, (3) kích thước và độ ổn định. Mình không ghi con số tốc độ dự kiến vì nó phụ thuộc phiên bản onnxruntime, số luồng và nhiệt độ Pi. Bước benchmark ở mục 10.4 sẽ cho số thật.

Kích thước ảnh đầu vào: thử 224, 320 và 384. Khởi đầu với **320** vì vết rách nhỏ cần phân giải cao hơn 224, chụp từ trên xuống nên hộp chiếm phần lớn khung hình.

### 7.3 Tiền xử lý và suy luận trên thiết bị

Tiền xử lý lúc chạy thật phải giống hệt lúc train, đây là lỗi hay gặp nhất khi chuyển model sang Pi:

1. Cắt ROI cố định (hình vuông phủ vùng hộp) từ khung chụp, thay đổi kích thước thẳng về 320 x 320.
2. BGR sang RGB, chia 255.
3. YOLOv8-cls: không trừ mean, không chia std. MobileNetV3 (torchvision): trừ mean và chia std của ImageNet. Mean và std ghi vào `model_card.json`, code đọc từ đó, không hard-code.
4. Layout NCHW, float32.
5. Chạy trên 3 khung, lấy trung bình xác suất để ổn định.

```python
import json, numpy as np, cv2, onnxruntime as ort

def to_probs(v):
    v = np.asarray(v, np.float32)
    if v.min() >= 0 and abs(v.sum() - 1) < 1e-3:   # YOLOv8-cls da softmax san
        return v
    e = np.exp(v - v.max())
    return e / e.sum()

class DamageClassifier:
    def __init__(self, onnx_path, card_path, threads=4):
        self.card = json.load(open(card_path))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name
        self.size = self.card['imgsz']
        self.mean = np.array(self.card['mean'], np.float32)
        self.std = np.array(self.card['std'], np.float32)
        self.classes = self.card['classes']

    def _prep(self, bgr):
        img = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(2, 0, 1)[None]

    def predict(self, frames):
        ps = [to_probs(self.sess.run(None, {self.inp: self._prep(f)})[0][0]) for f in frames]
        p = np.mean(ps, axis=0)
        return dict(zip(self.classes, p.tolist()))
```

Kiểm tra tương đương bắt buộc sau mỗi lần export: chạy 20 ảnh qua PyTorch và qua đoạn code trên, độ lệch xác suất lớn nhất phải dưới 1e-3, nếu không thì tiền xử lý đang sai.

## 8. Dữ liệu (phần quan trọng nhất)

Phần lớn chất lượng cuối cùng của hệ thống phụ thuộc vào dữ liệu, không phải vào việc chọn YOLO hay MobileNet. Dành thời gian cho mục này.

### 8.1 Năm nguyên tắc

1. **Chụp đúng môi trường triển khai.** Cùng camera, độ cao, đèn, nền băng chuyền, ROI. Dataset trên mạng chỉ để bổ trợ.
2. **Chia train/val/test theo hộp vật lý (`box_id`), không chia theo ảnh.** Nếu chia ngẫu nhiên theo ảnh, các ảnh của cùng một hộp rơi vào cả train lẫn test, accuracy sẽ đẹp giả và hỏng khi chạy thật.
3. **Số hộp khác nhau quan trọng hơn số ảnh mỗi hộp.** 12 ảnh của 100 hộp tốt hơn 100 ảnh của 12 hộp.
4. **Cân bằng lớp và có đủ hard negative** (hộp `ok` trông giống lỗi: nhăn, băng keo, bóng đổ, chữ in).
5. **Mọi ảnh đều truy vết được**: nguồn, phiên chụp, người gán nhãn, giấy phép (xem manifest ở 8.4).

### 8.2 Các nguồn dữ liệu

**Nguồn A: tự chụp trên băng chuyền (chính, bắt buộc).** Toàn bộ val và test phải đến từ nguồn này. Kế hoạch chi tiết ở 8.3.

**Nguồn B: dataset công khai (bổ trợ).** Mình đã tra thử trên Roboflow Universe, các bộ dưới đây đều ghi giấy phép CC BY 4.0 khi mình xem (cần ghi nguồn khi dùng). Trước khi dùng, bạn tự mở trang kiểm tra lại giấy phép, số ảnh và các lớp vì chủ dataset có thể đổi:

| Dataset | Dạng | Ghi chú |
| --- | --- | --- |
| "damaged boxes detection" trong project smart-damage-detection-for-logistics-packages-using-computer-vision (universe.roboflow.com/smart-damage-detection-for-logistics-packages-using-computer-vision/damaged-boxes-detection-imbqp) | Detection | Ảnh minh họa cho thấy các nhãn kiểu tear, leakage, squeeze, gần với `rach`, `uot`, `mop` nhất. Cần xem lại danh sách lớp thật trên trang |
| "Parcel Box Damage Classification" (universe.roboflow.com/project-hffml/parcel-box-damage-classification) | Classification | Trang liệt kê thêm các project liên quan như parcel-damages (812 ảnh) và defect (1506 ảnh) |
| "BoxCondition" (universe.roboflow.com/robot-arm-xlohb/boxcondition) | Classification nhị phân | damagedpackages và undamagedpackages |
| "Damaged Package Detection" (universe.roboflow.com/ameesh/damaged-package-detection-xjkol) | Classification | Dùng để lấy thêm ảnh hư hại chung |
| "damaged\_box" (universe.roboflow.com/bekarys0504-gmail-com/damaged\_box) | Detection | damaged và non-damaged |
| Hugging Face cvtechniques/PackageDamage | Mô hình kèm mô tả | README cho biết dataset gộp từ hai dataset Roboflow, khoảng 2.8 nghìn ảnh, 2 lớp, nhiều hộp carton chụp trong ánh sáng khá đồng đều. Có ghi chú model hay nhầm khi hộp chỉ nhăn nhẹ |

Từ khóa để tự tìm thêm: trên Roboflow Universe và Kaggle thử "damaged box", "parcel damage", "package defect", "cardboard tear", "cardboard dent". Nhìn chung dataset công khai chủ yếu là nhị phân (hỏng hoặc không), chụp bằng điện thoại, nền và ánh sáng khác băng chuyền của bạn. Cách dùng đúng:

- Dùng làm khởi tạo: train một pha ngắn trên dữ liệu ngoài rồi fine-tune trên dữ liệu tự chụp. Hoặc trộn vào train với tỉ lệ tối đa khoảng 30% mỗi lớp.
- **Tuyệt đối không đưa dữ liệu ngoài vào val và test.**
- Nhãn ngoài phải map về 5 lớp và rà bằng mắt. Ảnh "damaged" chung chung mà không phân biệt được lớp thì bỏ, đừng đoán nhãn.
- Ghi nguồn và giấy phép từng bộ vào `DATA_SOURCES.md` trong thư mục `external/`. Báo cáo cuối phải có phần trích dẫn.

**Nguồn C: dữ liệu phát sinh khi chạy thật (feedback).** Ảnh bị `AI_UNCERTAIN` hoặc ảnh người duyệt sửa nhãn trong Console được xuất về `data/feedback/` trên Drive, gán nhãn lại rồi vào dataset phiên bản kế tiếp. Đây là vòng cải thiện model theo thời gian.

**Nguồn D: ảnh tổng hợp.** Dán texture vết rách, chỉnh sửa ảnh. Hiệu quả hạn chế và rủi ro lệch phân phối, chỉ dùng cho train, làm sau cùng nếu còn thời gian.

**Không khuyến nghị:** cào ảnh Google Images hay mạng xã hội về train (vướng bản quyền, nhãn nhiễu).

### 8.3 Kế hoạch tự chụp

Mục tiêu phiên bản 1: khoảng 1700 ảnh thật.

| Lớp | Số hộp vật lý | Ảnh mỗi hộp | Tổng ảnh | Cách tạo mẫu |
| --- | --- | --- | --- | --- |
| ok | 40 | 12 | 480 | Hộp nguyên, thêm biến thể nhăn nhẹ, băng keo, nhãn, chữ in |
| rach | 30 | 12 | 360 | Xé mép nắp, rạch bìa bằng dao rọc (không thủng xuyên), bóc băng keo |
| mop | 30 | 12 | 360 | Đè, đạp, bóp góc và cạnh, ép bẹp một mặt |
| vo | 25 | 12 | 300 | Cắt khoét lỗ, làm nát góc, sập nắp, lộ hàng |
| uot | 20 | 12 | 240 | Phun nước làm sẫm bìa, vệt nước, bôi bùn hoặc cà phê loãng |

Một hộp vật lý có thể dùng cho nhiều lớp (chụp `ok` trước, làm rách rồi chụp `rach`). Khi đó các ảnh vẫn mang cùng `box_id` và sẽ nằm trong cùng một split. Tổng số hộp gốc cần khoảng 60 đến 80 hộp. Hộp tốt nhất là hộp giao hàng thương mại điện tử người ta bỏ đi (có sẵn vết hỏng thật, nhiều kích cỡ, màu sắc, in logo). Gom thêm ở quán tạp hóa, ký túc xá, nơi bán đồ trực tuyến. Mục tiêu đa dạng: ba cỡ (khoảng 15, 30, 45 cm), hộp nâu, hộp trắng, hộp in hình.

An toàn: dùng dao rọc cẩn thận, đeo găng tay, trải bạt khi làm ướt.

**Mức độ hư hại phải đa dạng:** vết rách nhỏ đến lớn (2 đến 15 cm), ở góc, ở mép, ở giữa mặt. Móp nhẹ, vừa, nặng. Riêng móp quá nhẹ đến mức hai người nhìn vẫn cãi nhau thì **không đưa vào train**, vì chỉ làm nhiễu nhãn.

**Biến thể chụp mỗi hộp (12 ảnh):** 4 hướng xoay (0, 90, 180, 270 độ) nhân 3 biến thể (lệch trái, giữa, lệch phải khoảng 2 cm, hoặc mức sáng khác nhau). Xoay vật lý trong lúc chụp tốt hơn xoay bằng phần mềm.

**Phiên chụp:** chia thành ít nhất 3 phiên trong ít nhất 2 ngày khác nhau để có biến thiên ánh sáng tự nhiên. **Dành riêng một phiên chụp (với hộp mới) chỉ làm test**, mô phỏng đúng tình huống triển khai.

**Thông số chụp cố định:**

- Độ phân giải 1280 x 960 hoặc 1536 x 864 (lưu toàn khung, ROI cắt ở bước sau để sau này đổi ROI vẫn dùng lại được).
- JPEG chất lượng 95.
- Khóa phơi sáng và cân bằng trắng (tắt auto) để màu và độ sáng ổn định. Với Picamera2: `AeEnable=False`, `ExposureTime`, `AnalogueGain`, `AwbEnable=False`, `ColourGains`, `AfMode=0`, `LensPosition`. Với webcam trên laptop: đặt thủ công bằng OpenCV nếu driver hỗ trợ, nếu không thì chấp nhận và ghi chú.
- Ghi lại trong `session.json`: ngày giờ, độ cao camera, mức đèn, người chụp, thiết bị.

Dùng script `tools/capture_dataset.py` để chụp (chạy được trên laptop với webcam và trên Pi): bấm phím số 0 đến 4 chọn lớp, nhập `box_id`, bấm Space để chụp, script tự đặt tên file, lưu vào thư mục phiên và ghi một dòng vào manifest. Nhập nhãn ngay lúc chụp, vì khi đó bạn biết chắc hộp thuộc lớp nào.

### 8.4 Cấu trúc Google Drive, quy ước đặt tên, manifest

```
MyDrive/parcel_sorter/
  data/
    raw/
      session_20261105_A/             # ảnh gốc theo phiên, không sửa
        B012_rach_s1105A_007.jpg
        session.json
    external/
      roboflow_boxcondition/          # kèm LICENSE.txt, SOURCE.txt
      roboflow_damaged_boxes_imbqp/
    feedback/                         # ảnh Console xuất về để gán nhãn lại
    labeled/
      v1/{ok,rach,mop,vo,uot}/        # ảnh đã duyệt nhãn
    manifests/
      dataset_v1.csv                  # bất biến; đổi thì tạo v2
    archives/
      dataset_v1_splits.zip           # train/ val/ test/ ở gốc zip
  models/
    yolov8n-cls_v1_320/               # best.pt, results.csv, confusion_matrix
    exports/                          # model_v1_20261120.onnx, model_card.json
  notebooks/  01_prepare.ipynb  02_train.ipynb  03_eval_export.ipynb
  docs/       label_guide.pdf  DATA_SOURCES.md
```

Quy ước tên file: `{box_id}_{lop}_{phien}_{stt:03d}.jpg`, ví dụ `B012_rach_s1105A_007.jpg`.

Manifest `dataset_vN.csv`, mỗi dòng một ảnh:

| Cột | Ý nghĩa |
| --- | --- |
| `file_path`, `sha256` | Đường dẫn tương đối và mã băm (phát hiện ảnh trùng và hỏng) |
| `box_id` | Mã hộp vật lý (dùng để chia split) |
| `class` | Một trong ok, rach, mop, vo, uot |
| `source`, `source_detail` | own, roboflow, kaggle, feedback và tên dataset cụ thể |
| `session`, `camera_height_cm`, `light_level`, `captured_at` | Điều kiện chụp |
| `labeler`, `reviewed_by` | Người gán và người duyệt |
| `license` | Giấy phép (bắt buộc với dữ liệu ngoài) |
| `split` | train, val hoặc test (script điền, không sửa tay) |

### 8.5 Gán nhãn và làm sạch

- Nhãn là tên thư mục. Không cần công cụ phức tạp. Nếu muốn giao diện thì dùng Label Studio chạy local hoặc Roboflow.
- Viết hướng dẫn gán nhãn 1 trang (bảng ở 7.1 kèm 2 đến 3 ảnh mẫu mỗi lớp) để mọi người cùng gán thống nhất.
- Người thứ hai duyệt ngẫu nhiên 10 đến 15% ảnh mỗi lớp. Nếu bất đồng trên 5% thì sửa lại hướng dẫn gán nhãn rồi duyệt lại.
- Làm sạch: bỏ ảnh nhòe, bỏ ảnh mất vùng hộp, bỏ ảnh trùng (dùng thư viện `imagehash` so perceptual hash, hoặc so `sha256`).

### 8.6 Chia tập train/val/test

- Tỉ lệ khoảng 70/15/15, **chia theo `box_id`**, phân tầng theo lớp, seed cố định (ghi seed vào `configs/split_seed.txt`).
- Phiên chụp dành riêng làm test thì ưu tiên gán vào test trước, sau đó chia phần còn lại.
- Dữ liệu ngoài và `feedback` chỉ vào train.
- Mất cân bằng lớp: oversample (nhân bản ảnh lớp ít) chỉ ở bước tạo thư mục train, không làm với val và test.

```python
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv('manifests/dataset_v1.csv')
own = df[df.source == 'own'].copy()
own['split'] = 'train'
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)  # moi fold khoang 14%
for i, (_, idx) in enumerate(sgkf.split(own, own['class'], groups=own['box_id'])):
    if i == 0:
        own.iloc[idx, own.columns.get_loc('split')] = 'test'
    elif i == 1:
        own.iloc[idx, own.columns.get_loc('split')] = 'val'
df.loc[df.source != 'own', 'split'] = 'train'
df.update(own)
# sau do script copy anh vao splits/v1/{train,val,test}/<lop>/ roi nen zip
```

Sau khi chia, bắt buộc kiểm tra: không `box_id` nào xuất hiện ở hai split.

### 8.7 Tăng cường dữ liệu (augmentation, chỉ cho train)

| Nên dùng | Không nên dùng |
| --- | --- |
| Lật ngang và dọc, xoay 90 độ, xoay nhẹ ±10 độ | Cắt ngẫu nhiên mạnh (có thể cắt mất vùng hỏng làm nhãn sai) |
| Độ sáng và tương phản ±25% | Đổi màu mạnh (làm hỏng dấu hiệu `uot` là bìa sẫm màu) |
| Hue nhẹ (khoảng ±0.01), bão hòa ±30% | Random erasing (có thể xóa đúng vết hỏng) |
| Làm mờ nhẹ, nhiễu nhẹ, dịch và scale nhẹ (0.9 đến 1.1) | Mixup, CutMix |

## 9. Train trên Google Colab (từng bước)

### 9.1 Chuẩn bị một lần

1. Tạo cấu trúc Drive như 8.4 và upload `dataset_v1_splits.zip` vào `data/archives/`.
2. Mở Colab, chọn Runtime, Change runtime type, GPU (T4 là đủ).
3. Colab bản miễn phí có thể ngắt phiên bất ngờ, nên mọi checkpoint lưu ra Drive và dùng `resume=True` khi cần.
4. **Đừng train đọc trực tiếp từ Drive**, rất chậm. Luôn copy zip về đĩa cục bộ của Colab rồi giải nén.

### 9.2 Các cell của notebook `02_train.ipynb`

**Cell 1: mount Drive và kiểm tra GPU**

```python
from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi
```

**Cell 2: cài thư viện**

```python
!pip -q install ultralytics onnx onnxruntime scikit-learn
```

**Cell 3: copy dữ liệu về đĩa cục bộ**

```python
DRIVE = '/content/drive/MyDrive/parcel_sorter'
!mkdir -p /content/ds
!cp {DRIVE}/data/archives/dataset_v1_splits.zip /content/ds/
!cd /content/ds && unzip -q dataset_v1_splits.zip
```

**Cell 4: đếm ảnh mỗi tập và mỗi lớp (phải khớp kế hoạch, và các lớp không lệch quá nhiều)**

```python
import os
for s in ['train', 'val', 'test']:
    print(s, {c: len(os.listdir(f'/content/ds/{s}/{c}')) for c in sorted(os.listdir(f'/content/ds/{s}'))})
```

**Cell 5: train YOLOv8n-cls**

```python
from ultralytics import YOLO
model = YOLO('yolov8n-cls.pt')
model.train(data='/content/ds', imgsz=320, epochs=60, batch=32, patience=15,
            project=f'{DRIVE}/models', name='yolov8n-cls_v1_320', exist_ok=True,
            seed=42, workers=2, save_period=5,
            scale=0.1, fliplr=0.5, flipud=0.5, erasing=0.0,
            hsv_h=0.01, hsv_s=0.3, hsv_v=0.3)
# neu phien bi ngat: YOLO(f'{DRIVE}/models/yolov8n-cls_v1_320/weights/last.pt').train(resume=True)
```

Tên tham số augmentation có thể khác giữa các phiên bản Ultralytics. Chạy `yolo cfg` hoặc đọc tài liệu của phiên bản đang cài để xác nhận, mục tiêu là: không cắt ngẫu nhiên mạnh, tắt random erasing.

**Cell 6: đánh giá trên test**

```python
import glob, numpy as np
from sklearn.metrics import classification_report, confusion_matrix

best = YOLO(f'{DRIVE}/models/yolov8n-cls_v1_320/weights/best.pt')
names = [best.names[i] for i in range(len(best.names))]   # thu tu alphabet theo thu muc, KHONG hard-code
y_true, y_pred, y_prob = [], [], []
for ci, c in enumerate(names):
    for p in glob.glob(f'/content/ds/test/{c}/*.jpg'):
        r = best.predict(p, imgsz=320, verbose=False)[0]
        y_true.append(ci); y_pred.append(int(r.probs.top1)); y_prob.append(r.probs.data.cpu().numpy())
print(classification_report(y_true, y_pred, target_names=names, digits=3))
print(confusion_matrix(y_true, y_pred))

ok = names.index('ok')
t_dmg = np.array(y_true) != ok
p_dmg = np.array(y_pred) != ok
print('damage recall :', (t_dmg & p_dmg).sum() / t_dmg.sum())
print('false reject  :', (~t_dmg & p_dmg).sum() / (~t_dmg).sum())
```

Lưu ý quan trọng: thứ tự lớp trong model là thứ tự alphabet của tên thư mục (mop, ok, rach, uot, vo), khác với thứ tự bạn nghĩ (ok, rach, mop, vo, uot). Luôn đọc `names` từ model, ghi vào `model_card.json`, đừng hard-code trong code của Pi.

**Cell 7: export ONNX và lưu vào Drive**

```python
import shutil
onnx_path = best.export(format='onnx', imgsz=320, opset=12, simplify=True, dynamic=False)
shutil.copy(onnx_path, f'{DRIVE}/models/exports/model_v1_20261120.onnx')
```

Sau đó tạo `model_card.json` (mẫu ở 10.5) cùng thư mục.

### 9.3 Baseline MobileNetV3-Small bằng PyTorch (rút gọn)

```python
import torch, torch.nn as nn
from torchvision import datasets, transforms, models

tf_tr = transforms.Compose([transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.25, 0.25, 0.3, 0.01), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
tf_ev = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train = datasets.ImageFolder('/content/ds/train', tf_tr)
val = datasets.ImageFolder('/content/ds/val', tf_ev)

m = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(train.classes))
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
# Pha 1 (5 epoch): dong bang m.features, lr 1e-3 cho classifier
# Pha 2 (20-30 epoch): mo khoa tat ca, lr 1e-4, AdamW, early stop theo val damage-recall
# Export: torch.onnx.export(m.eval(), torch.randn(1, 3, 320, 320), 'mnv3s.onnx',
#         opset_version=13, input_names=['images'], output_names=['logits'])
```

Output ONNX của MobileNetV3 là logits, hàm `to_probs` ở 7.3 đã xử lý softmax.

### 9.4 Các thí nghiệm cần chạy và ghi lại

Thay đổi một yếu tố mỗi lần, ghi kết quả vào `experiments.csv` (tên thí nghiệm, kích thước ảnh, model, dữ liệu, accuracy, damage recall, false reject, độ trễ trên Pi nếu có).

| Mã | Nội dung | Mục đích |
| --- | --- | --- |
| E1 | YOLOv8n-cls, 224, chỉ dữ liệu tự chụp | Mốc ban đầu |
| E2 | YOLOv8n-cls, 320 | Xem phân giải có giúp nhận vết nhỏ |
| E3 | MobileNetV3-Small, 320 | So sánh kiến trúc cùng điều kiện |
| E4 | E2 + dữ liệu ngoài (tối đa 30%) | Dữ liệu ngoài có thật sự giúp không |
| E5 | E2 với 2 hạt giống khác nhau | Đo độ dao động giữa các lần train |

### 9.5 Đọc kết quả và xử lý khi chưa đạt

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
| --- | --- | --- |
| Train rất cao, val thấp | Ít hộp, model học thuộc hộp | Thu thêm hộp khác nhau, tăng augmentation nhẹ |
| `rach` nhầm `vo` | Ranh giới nhãn không rõ | Viết lại hướng dẫn gán nhãn, rà lại ảnh nhập nhằng |
| `mop` nhầm `ok` | Móp nhẹ quá, ánh sáng phẳng | Bỏ ảnh móp quá nhẹ khỏi train, đổi góc đèn để hiện bóng đổ |
| `uot` nhầm `ok` | Ít mẫu, màu ướt không nổi | Thêm mẫu ướt ở nhiều mức, khóa cân bằng trắng |
| Test thấp hơn val nhiều | Phiên test khác điều kiện | Thêm phiên chụp ở điều kiện đa dạng vào train |
| Hiệu năng tốt trên laptop, tệ trên Pi | Tiền xử lý lệch, camera khác | So sánh ảnh đầu vào giữa laptop và Pi, kiểm tra ROI và phơi sáng |

## 10. Đánh giá, chọn ngưỡng, xuất và benchmark

### 10.1 Chỉ số báo cáo

- Ma trận nhầm lẫn (confusion matrix) và precision, recall, F1 từng lớp.
- **Recall phát hiện hư hại** (gộp bốn lớp lỗi so với `ok`): chỉ số quan trọng nhất vì bỏ sót hộp hỏng là lỗi nặng nhất.
- **False reject**: tỉ lệ hộp tốt bị loại nhầm.
- Độ trễ trên Pi (p50, p95) và kích thước model.

### 10.2 Chọn ngưỡng trên tập validation

Quét ngưỡng tin cậy (`tau_conf`) từ 0.4 đến 0.9, xem đánh đổi giữa bỏ sót và loại nhầm, chọn ngưỡng thỏa KPI. Chọn ngưỡng trên **val**, chỉ báo cáo số trên **test** một lần, không chỉnh theo test.

```python
def sweep(y_true_dmg, probs, ok_idx, taus):
    top, conf = probs.argmax(1), probs.max(1)
    for t in taus:
        reject = (top != ok_idx) | (conf < t)      # AI_UNCERTAIN cung bi loai
        miss = (y_true_dmg & ~reject).sum() / y_true_dmg.sum()
        false_rej = (~y_true_dmg & reject).sum() / (~y_true_dmg).sum()
        print(f'tau={t:.2f}  miss={miss:.3f}  false_reject={false_rej:.3f}')
```

### 10.3 Xuất ONNX và (tùy chọn) lượng tử hóa INT8

- Luôn kiểm tra tương đương PyTorch và ONNX (7.3) sau khi export.
- INT8: dùng lượng tử hóa tĩnh của ONNX Runtime với khoảng 200 ảnh train làm tập hiệu chuẩn. Chấp nhận bản INT8 chỉ khi recall hư hại giảm không quá 1 đến 2 điểm phần trăm so với FP32 **và** đo được lợi ích tốc độ thật trên Pi.

### 10.4 Benchmark trên laptop rồi trên Pi

Script `ml/benchmark.py`: khởi động 10 lần chạy bỏ đi, đo 100 lần, in p50 và p95, thử 1, 2, 4 luồng. Trên Pi in thêm nhiệt độ (`vcgencmd measure_temp`) và cờ throttle (`vcgencmd get_throttled`, kết quả `0x0` là ổn). Chạy trên laptop trước để có mốc, rồi chạy trên Pi.

| Model | Kích thước ảnh | Luồng | p50 laptop | p95 laptop | p50 Pi | p95 Pi | Recall hư hại |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (điền sau khi đo) |  |  |  |  |  |  |  |

### 10.5 Phiên bản hóa model: `model_card.json`

```json
{
  "name": "yolov8n-cls_v1_320",
  "version": "v1-20261120",
  "classes": ["mop", "ok", "rach", "uot", "vo"],
  "imgsz": 320,
  "mean": [0, 0, 0],
  "std": [1, 1, 1],
  "output": "probabilities",
  "dataset_version": "v1",
  "metrics": {"acc_test": null, "damage_recall": null, "false_reject": null, "pi4_p95_ms": null},
  "thresholds": {"tau_conf": 0.6, "tau_damage": 0.5},
  "sha256": "<ma bam cua file onnx>"
}
```

Các giá trị `null` điền sau khi đo. Pi chỉ nạp model khi `sha256` khớp.

## 11. Chiến lược: test trên laptop trước, đẩy lên Pi sau

### 11.1 Nguyên tắc

Một code base duy nhất. Khác biệt giữa laptop và Pi chỉ nằm ở file cấu hình (`config/laptop.yaml` và `config/pi.yaml`) và ở các lớp phần cứng được trừu tượng hóa thành interface:

| Interface | Trên laptop | Trên Pi |
| --- | --- | --- |
| Camera | `OpenCVCamera` (webcam) hoặc `FolderCamera` (phát lại thư mục ảnh/video) | `PiCamera2Camera` |
| Controller (nói chuyện với ESP32) | `MockController` (giả lập: phát DETECT sau n giây, trả ACK/DONE) hoặc `SerialController` qua cáp USB với ESP32 thật | `SerialController('/dev/serial0')` |
| Trigger | Phím Space hoặc timer | EVT DETECT từ ESP32 |
| MQTT | Mosquitto hoặc ThingsBoard chạy Docker local | ThingsBoard thật qua TLS 8883 |
| Console API | uvicorn trên localhost | Console trên máy dev qua HTTPS trong LAN |
| Model | Cùng file ONNX | Cùng file ONNX (hoặc bản INT8 nếu chọn) |

Lưu ý về dữ liệu: ở giai đoạn laptop, bạn chỉ cần **dataset v0** nhỏ (khoảng 40 ảnh mỗi lớp, chụp bằng webcam hoặc điện thoại) để chạy thử toàn bộ pipeline từ train đến suy luận. **Dataset v1** để train model cuối phải chụp trên chính giàn băng chuyền với camera và đèn cố định (mục 8.3). Đừng lấy kết quả của model v0 để kết luận về độ chính xác.

### 11.2 Cấu trúc repo

```
parcel-sorter/
  README.md
  ml/                      make_splits.py  benchmark.py  eval_onnx.py  configs/
  pi/
    sorter/
      main.py              # vong lap state machine
      config.py            # doc YAML bang pydantic
      camera.py            # Camera, OpenCVCamera, PiCamera2Camera, FolderCamera
      controller.py        # Controller, SerialController, MockController, crc16
      qr.py                # giai ma + kiem HMAC
      ai.py                # DamageClassifier
      decision.py          # bang quyet dinh (ham thuan, de unit test)
      storage.py           # SQLite: seen_parcels, events, outbox
      uplink.py            # MQTT + HTTP, store-and-forward
      models/              # model.onnx, model_card.json
    config/                laptop.yaml  pi.yaml
    tests/
  esp32/                   # du an PlatformIO
  console/                 # FastAPI app (muc 13)
  tools/                   capture_dataset.py  make_qr_labels.py  ota_tool/
  deploy/                  docker-compose.yml  sorter.service
  docs/
```

### 11.3 Môi trường dev trên laptop

- Python 3.11, tạo venv: `python -m venv .venv`.
- Thư viện: `opencv-python`, `numpy`, `onnxruntime`, `pyzbar`, `paho-mqtt`, `pyserial`, `pydantic`, `pyyaml`, `httpx`, `pytest`. `picamera2` chỉ cài trên Pi.
- `pyzbar` trên Windows đã kèm DLL zbar trong wheel. Trên Linux cần `sudo apt install libzbar0`.
- Docker để chạy ThingsBoard và Mosquitto (`deploy/docker-compose.yml`).
- Firmware ESP32 dùng cờ build `LINK_UART=0` (UART0 qua USB) khi test với laptop và `LINK_UART=2` (chân GPIO16/17) khi nối với Pi.

### 11.4 Giai đoạn L (laptop): các bước và cổng chuyển

| Bước | Việc | Cách kiểm chứng |
| --- | --- | --- |
| L0 | Dựng repo, venv, CI chạy `pytest` | `pytest` chạy được, kể cả khi chưa có code thật |
| L1 | Dataset v0, train và export bằng Colab theo mục 9, nhận file ONNX và model card | So khớp PyTorch và ONNX (mục 7.3) |
| L2 | `qr.py` và sinh QR (`make_qr_labels.py`): ký, giải mã, kiểm HMAC, hạn, trùng | Unit test với QR đúng, sai MAC, hết hạn, trùng, sai định dạng; thử với QR in ra giấy và webcam |
| L3 | `ai.py` + `benchmark.py` | Có bảng độ trễ trên laptop làm mốc |
| L4 | `decision.py` | Unit test bảng quyết định, mỗi dòng của mục 4.2 ít nhất một test |
| L5 | `main.py` + `MockController` + `FolderCamera` | Phát lại 50 ảnh, kết quả khay và mã lý do đúng với kỳ vọng ghi sẵn |
| L6 | `storage.py`, `uplink.py`, Console khung sườn, ThingsBoard Docker | Sự kiện hiện trên dashboard và Console; rút mạng rồi cắm lại thì outbox đẩy đủ, không trùng |
| L7 | Nối ESP32 thật qua USB, chạy `SerialController` | PING/PONG, CRC sai bị NAK, SORT điều khiển được servo, watchdog hoạt động |

**Cổng chuyển sang Pi:** toàn bộ unit test xanh; L5 đạt 100% trên bộ 50 ảnh; luồng uplink chạy đúng; firmware ESP32 qua được L7. Chưa qua cổng thì chưa mang lên Pi, vì sửa lỗi logic trên Pi mất công gấp nhiều lần.

### 11.5 Bộ test case

Laptop: chạy với mock hoặc ESP32 qua USB. Pi: chạy lại đúng bộ này trên giàn thật.

| Mã | Kịch bản | Kỳ vọng | Laptop | Pi |
| --- | --- | --- | --- | --- |
| TC-01 | QR hợp lệ kho A, hộp ok | Khay A, mã OK\_A | Có | Có |
| TC-02 | QR hợp lệ kho B, hộp ok | Khay B, mã OK\_B | Có | Có |
| TC-03 | QR hợp lệ, lần lượt hộp rach, mop, vo, uot | Khay Loại, mã DAMAGE\_x tương ứng (4 ca) | Có | Có |
| TC-04 | Che QR | Khay Loại, QR\_UNREADABLE | Có | Có |
| TC-05 | QR sửa 1 ký tự | Khay Loại, QR\_INVALID, cảnh báo | Có | Có |
| TC-06 | QR quá hạn | Khay Loại, QR\_EXPIRED | Có | Có |
| TC-07 | Quét lại cùng mã | Khay Loại, QR\_REPLAY, cảnh báo | Có | Có |
| TC-08 | Kho C | Khay Loại, WAREHOUSE\_UNKNOWN | Có | Có |
| TC-09 | Ảnh mờ hoặc vật lạ | Khay Loại, AI\_UNCERTAIN | Có | Có |
| TC-10 | Rút dây UART | ESP32 dừng băng trong khoảng 3 s, dashboard báo lỗi | Có (USB) | Có |
| TC-11 | Mất mạng giữa chừng | Vẫn phân loại; có mạng lại thì outbox đẩy đủ, không trùng | Có | Có |
| TC-12 | Chèn nhiễu làm CRC sai | NAK, không gạt | Có | Có |
| TC-13 | Gửi lại lệnh UART với SEQ cũ | Bị từ chối | Có | Có |
| TC-14 | Bấm E-Stop | Motor dừng, EVT ESTOP, dashboard cảnh báo | Không | Có |
| TC-15 | 100 kiện liên tục | Không treo, RAM ổn định | Có (mock) | Có |
| TC-16 | RPC ngoài whitelist | Bị từ chối, có ghi log | Có | Có |
| TC-17 | Model có sha256 sai | Pi từ chối nạp, giữ model cũ | Có | Có |
| TC-18 | Bật tắt đèn phòng khi chạy | Độ chính xác giảm không đáng kể (đo trên ít nhất 30 hộp) | Không | Có |

### 11.6 Giai đoạn P (Pi): các bước

1. **Cài hệ điều hành.** Raspberry Pi OS 64-bit (Bookworm), cập nhật, đặt hostname, bật SSH. Chạy `sudo raspi-config`, Interface Options, Serial Port: tắt login shell qua serial, bật serial hardware. Kiểm tra có `/dev/serial0`.
2. **Cài gói hệ thống.** `sudo apt install -y python3-picamera2 libzbar0 git`. Tạo venv bằng `python3 -m venv --system-site-packages /opt/parcel-sorter/venv` để dùng được picamera2. Trong venv: `pip install onnxruntime opencv-python-headless numpy pyzbar paho-mqtt pyserial pydantic pyyaml httpx`.
3. **Đồng bộ code và model.** `git clone` repo; copy `model.onnx` và `model_card.json` từ Drive vào `pi/sorter/models/` (bằng `scp` hoặc `rclone`). Kiểm tra `sha256` khớp model card.
4. **Benchmark.** Chạy `ml/benchmark.py` trên Pi, điền bảng 10.4, quyết định model cuối.
5. **Camera.** Chụp thử, canh ROI, khóa phơi sáng và lấy nét. Nếu chưa có dataset v1 thì chụp bằng `capture_dataset.py` ngay bước này.
6. **UART.** Thử PING/PONG và CRC với ESP32 thật.
7. **Hiệu chuẩn cơ khí.** Tốc độ băng, `STOP_DELAY_MS`, `T_TRAVEL_MS`, góc servo, ngưỡng cảm biến (đo giá trị cảm biến khi có và không có hộp rồi đặt ngưỡng ở giữa).
8. **Chạy toàn bộ TC-01 đến TC-18** với hộp thật.
9. **Cài dịch vụ systemd** để tự khởi động, bật log rotate.
10. **Soak test.** 30 phút hoặc 100 kiện, ghi nhiệt độ CPU và cờ throttle. Đạt KPI mục 1.3 mới coi xong giai đoạn P.

```ini
[Unit]
Description=Parcel Sorter Edge App
After=network-online.target
Wants=network-online.target

[Service]
User=sorter
SupplementaryGroups=dialout video
WorkingDirectory=/opt/parcel-sorter/pi
ExecStart=/opt/parcel-sorter/venv/bin/python -m sorter.main --config config/pi.yaml
Restart=always
RestartSec=3
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/var/lib/sorter /var/log/sorter

[Install]
WantedBy=multi-user.target
```

## 12. Phần mềm edge trên Pi

### 12.1 Các module

| Module | Trách nhiệm |
| --- | --- |
| `camera.py` | Giao diện `grab_n(n, gap_ms)`, ba cài đặt (OpenCV, Picamera2, Folder). Khóa phơi sáng theo config |
| `controller.py` | Gửi và nhận khung UART, CRC, SEQ, retry, ACK; trả về sự kiện cho vòng chính |
| `qr.py` | Tiền xử lý, giải mã bằng pyzbar rồi OpenCV, kiểm HMAC, hạn, trùng, kho |
| `ai.py` | Nạp model, kiểm `sha256`, suy luận trên N khung, timeout |
| `decision.py` | Hàm thuần theo bảng 4.2 (không đụng I/O, dễ unit test) |
| `storage.py` | SQLite: bảng `seen_parcels`, `events`, `outbox`; dọn ảnh cũ |
| `uplink.py` | Đẩy MQTT (ThingsBoard) và HTTPS (Console), store-and-forward, heartbeat |
| `main.py` | Vòng lặp trạng thái, xử lý lỗi, tắt máy an toàn |

### 12.2 Cấu hình mẫu

```yaml
profile: pi                # laptop | pi
camera:
  kind: picamera2          # opencv | picamera2 | folder
  device: 0
  folder: ./replay_images
  width: 1280
  height: 960
  roi: [160, 0, 960, 960]  # x, y, w, h (vuong)
  manual: {exposure_us: 8000, gain: 1.5, lens_position: 4.0}
controller:
  kind: serial             # serial | mock
  port: /dev/serial0
  baud: 115200
  ack_timeout_ms: 200
  retries: 3
timing:
  settle_ms: 400
  frames: 3
  frame_gap_ms: 80
ai:
  model: models/model.onnx
  card: models/model_card.json
  threads: 4
  tau_conf: 0.60
  tau_damage: 0.50
policy: {ok: PASS, rach: REJECT, mop: REJECT, vo: REJECT, uot: REJECT}
qr:
  key_path: /etc/sorter/hmac.key
  max_age_hours: 72
  warehouses: [A, B]
uplink:
  mqtt: {host: tb.local, port: 8883, ca: /etc/sorter/ca.pem, cert: /etc/sorter/device.pem, key: /etc/sorter/device.key}
  console: {base_url: 'https://console.local:8443', api_key_path: /etc/sorter/console.key}
  outbox_max_mb: 500
storage:
  db_path: /var/lib/sorter/sorter.db
  image_dir: /var/lib/sorter/images
  keep_images_days: 14
```

### 12.3 `decision.py`: hàm thuần để test

```python
from dataclasses import dataclass

@dataclass
class Decision:
    tray: str           # 'A' | 'B' | 'R'
    reason: str         # ma ly do (muc 4.2)
    alert: bool = False # canh bao bao mat

def decide(qr_status, warehouse, probs, cfg, ai_error=False):
    if qr_status == 'QR_UNREADABLE':
        return Decision('R', qr_status)
    if qr_status in ('QR_INVALID', 'QR_REPLAY'):
        return Decision('R', qr_status, alert=True)
    if qr_status in ('QR_EXPIRED', 'WAREHOUSE_UNKNOWN'):
        return Decision('R', qr_status)
    if ai_error or not probs:
        return Decision('R', 'AI_ERROR')
    top = max(probs, key=probs.get)
    conf = probs[top]
    if top != 'ok' and conf >= cfg.tau_damage and cfg.policy[top] == 'REJECT':
        return Decision('R', 'DAMAGE_' + top.upper())
    if conf < cfg.tau_conf:
        return Decision('R', 'AI_UNCERTAIN')
    reason = 'OK_' + warehouse if top == 'ok' else 'PASS_FLAG_' + top.upper()
    return Decision(warehouse, reason)
```

### 12.4 Vòng lặp chính (rút gọn)

```python
while running:
    ev = ctrl.wait_event(timeout=1.0)           # DETECT / STOPPED / ESTOP / JAM
    if ev in ('ESTOP', 'JAM'):
        enter_error(ev); continue
    if ev != 'STOPPED':
        continue
    time.sleep(cfg.timing.settle_ms / 1000)
    frames = cam.grab_n(cfg.timing.frames, cfg.timing.frame_gap_ms)
    qr_status, info = qr.process(frames, store)
    probs, ai_err = ai.safe_predict([crop_roi(f) for f in frames])
    dec = decide(qr_status, info and info['warehouse'], probs, cfg.ai, ai_err)
    if not ctrl.send('SORT', dec.tray):         # co ACK + retry ben trong
        enter_error('UART'); continue
    ctrl.wait_event('DONE', timeout=5.0)
    if info: store.mark_seen(info['parcel_id'], dec.reason)
    event_id = store.log_event(info, probs, dec, frames[len(frames) // 2])
    uplink.enqueue(event_id)                    # chay nen, khong chan vong chinh
```

Chỉ ghi `seen_parcels` khi QR hợp lệ và đã có quyết định. Mã QR sai chữ ký không bao giờ được ghi vào bảng đó, nếu không kẻ tấn công có thể làm đầy bảng.

### 12.5 Hàng đợi offline (store-and-forward)

- Bảng `outbox(id, kind, payload_json, image_path, created_at, attempts, sent_at)`.
- Mỗi sự kiện có `event_uuid`. Console dùng nó làm khóa chống trùng (idempotent), nên gửi lại bao nhiêu lần cũng không bị nhân đôi.
- Luồng nền gửi theo thứ tự thời gian, lùi thời gian chờ tăng dần (exponential backoff), MQTT dùng QoS 1.
- Giới hạn dung lượng ảnh trong outbox (`outbox_max_mb`): khi đầy thì xóa ảnh cũ nhất nhưng giữ metadata.

### 12.6 Log và giám sát

- Log dạng JSON, xoay vòng (rotating), không ghi khóa bí mật vào log.
- Heartbeat 30 s: nhiệt độ CPU, cờ throttle, uptime, phiên bản app và model, kích thước outbox, trạng thái UART. Gửi lên ThingsBoard (thuộc tính thiết bị) và Console.

### 12.7 Firmware ESP32

- Tách thành các phần: bộ phân tích khung UART (CRC, SEQ, whitelist), điều khiển motor (tăng giảm tốc PWM mềm), cảm biến (chống dội 30 ms), điều khiển servo theo thời gian, máy trạng thái, watchdog, mô-đun OTA.
- Máy trạng thái: READY, khi cảm biến thấy hộp thì phát EVT DETECT, chờ `STOP_DELAY_MS` rồi dừng motor và phát EVT STOPPED. Nhận SORT thì chạy băng, chờ `T_TRAVEL_MS`, gạt servo giữ `SERVO_HOLD_MS`, trả servo về, phát EVT DONE, quay về READY. SORT,R (khay Loại) không gạt servo, hộp trôi đến cuối băng.
- Hằng số hiệu chỉnh đặt trong `config.h`: `STOP_DELAY_MS`, `T_TRAVEL_MS`, `SERVO_HOLD_MS`, `DEBOUNCE_MS`, `WDT_MS` (3000), `PWM_MAX`.
- Mọi tham số lệnh phải được kiểm tra khoảng giá trị trước khi dùng (ví dụ SPEED chỉ 0 đến PWM\_MAX).

## 13. Phần mềm quản lý (Management Console)

### 13.1 Mục tiêu và vai trò người dùng

ThingsBoard lo phần giám sát realtime. Console lo phần nghiệp vụ mà ThingsBoard làm không tiện: đăng ký kiện và in QR, lưu ảnh và hồ sơ từng sự kiện, duyệt ảnh lỗi để cải thiện model, quản lý model và cấu hình, báo cáo.

| Vai trò | Quyền |
| --- | --- |
| Admin | Mọi thứ: người dùng, cấu hình, model, firmware, mở khóa quét lại, xem audit |
| Operator | Tạo kiện, in QR, duyệt ảnh lỗi, xem sự kiện và thống kê |
| Viewer | Chỉ xem sự kiện và thống kê |

### 13.2 Chức năng

| Mã | Chức năng | Mức | Mô tả |
| --- | --- | --- | --- |
| M1 | Đăng nhập, phân quyền | Must | Mật khẩu băm argon2id, JWT trong cookie HttpOnly, khóa tài khoản sau 5 lần sai, 2FA TOTP (Should) |
| M2 | Quản lý kiện | Must | Tạo kiện đơn hoặc hàng loạt (nhập CSV: kho, mã tùy chọn), sinh QR có HMAC, xuất PDF nhãn, theo dõi trạng thái kiện, nút "Cho phép quét lại" |
| M3 | Sự kiện phân loại | Must | Bảng cập nhật tự động (HTMX polling 5 s), lọc theo thời gian, kết quả, lý do, kho, thiết bị. Trang chi tiết: ảnh, xác suất từng lớp, trạng thái QR, độ trễ, phiên bản model. Xuất CSV |
| M4 | Duyệt ảnh và phản hồi | Should | Hàng chờ gồm ảnh AI\_UNCERTAIN, ảnh DAMAGE có độ tin cậy thấp và 5% ảnh PASS lấy ngẫu nhiên. Người duyệt chọn nhãn đúng (phím tắt 1 đến 5). Nút "Xuất dataset": gom ảnh đã duyệt theo thư mục lớp kèm manifest CSV (cột như mục 8.4, `source=feedback`) thành file zip để tải về và up lên Drive |
| M5 | Thống kê và báo cáo | Must | Số lượng theo ngày, kho, lớp lỗi; tỉ lệ hư hại; thời gian xử lý trung bình và p95; số cảnh báo bảo mật; top lý do loại; độ đúng của AI so với nhãn người duyệt; xuất CSV |
| M6 | Thiết bị | Should | Danh sách Pi và ESP32: online hay offline (theo heartbeat), phiên bản app, model, firmware, nhiệt độ. Đăng ký thiết bị mới sinh API key (hiện một lần), thu hồi key |
| M7 | Quản lý model | Should | Upload `model.onnx` và `model_card.json`, kiểm `sha256`, đặt bản active, rollback. Pi định kỳ hỏi bản active, tải về, kiểm `sha256` rồi nạp khi rảnh |
| M8 | Cấu hình | Should | Ngưỡng `tau_conf`, `tau_damage`, chính sách theo lớp, `max_age_hours`, danh sách kho. Có phiên bản, Pi định kỳ hỏi cấu hình mới |
| M9 | Nhật ký kiểm toán | Must | Ai làm gì lúc nào: đăng nhập, đổi cấu hình, mở khóa quét lại, tải model, kích hoạt OTA. Chỉ thêm, không sửa, không xóa |
| M10 | Firmware và OTA | Should | Lưu các bản firmware đã ký cùng manifest, hiển thị lịch sử nạp (xem mục 16) |

Dùng chung một thư viện `shared/qrsign.py` (ký và kiểm QR) cho cả Pi, Console và công cụ in nhãn, để hai phía không bao giờ lệch định dạng.

### 13.3 Công nghệ đề xuất

- Backend: FastAPI, SQLModel (SQLAlchemy), SQLite chế độ WAL cho MVP, đổi sang PostgreSQL nếu cần.
- Giao diện: Jinja2 + HTMX + Bootstrap, không cần build frontend. Biểu đồ dùng Chart.js.
- Nhãn PDF: thư viện `qrcode`, `Pillow`, `reportlab`.
- Bảo mật: `argon2-cffi`, `pyjwt`, `pyotp`.
- Chạy bằng uvicorn đặt sau Caddy hoặc nginx, HTTPS bằng chứng chỉ nội bộ trong LAN.
- Ảnh lưu trên đĩa theo `images/YYYY/MM/DD/<event_uuid>.jpg`, DB chỉ lưu đường dẫn.
- Nếu cần ra sản phẩm nhanh hơn và chấp nhận phân quyền yếu, có thể dùng Streamlit, nhưng spec này giả định FastAPI.

### 13.4 Mô hình dữ liệu

| Bảng | Cột chính |
| --- | --- |
| `users` | id, username, pw\_hash, role, totp\_secret, totp\_enabled, failed\_logins, locked\_until, created\_at |
| `parcels` | id, parcel\_id (duy nhất), warehouse, created\_by, created\_at, qr\_ts, status, rescan\_allowed, notes |
| `devices` | id, name, kind, api\_key\_hash, enabled, last\_seen, app\_version, model\_version, fw\_version, cpu\_temp, throttled |
| `scan_events` | id, event\_uuid (duy nhất), device\_id, parcel\_id, ts\_device, ts\_received, qr\_status, warehouse, ai\_probs\_json, ai\_top, ai\_conf, tray, reason, alert, latency\_ms, model\_version, image\_path |
| `reviews` | id, event\_id, reviewer\_id, true\_label, comment, reviewed\_at, exported\_at |
| `models` | id, version, sha256, file\_path, card\_json, status (candidate, active, retired), uploaded\_by, uploaded\_at |
| `configs` | id, version, json, created\_by, created\_at, active |
| `firmware` | id, version, sha256, signature, manifest\_json, file\_path, uploaded\_at |
| `audit_logs` | id, ts, user\_id, action, target, detail\_json, ip |

Chỉ mục: `scan_events(ts)`, `scan_events(reason)`, `scan_events(parcel_id)`.

### 13.5 API giữa Pi và Console

Tất cả qua HTTPS, thiết bị xác thực bằng `Authorization: Bearer <api_key>` (Should: mTLS).

| Phương thức và đường dẫn | Mô tả |
| --- | --- |
| `POST /api/v1/events` | Multipart gồm `meta` (JSON) và `image` (JPEG). Trả 201, hoặc 200 nếu `event_uuid` đã có (idempotent) |
| `POST /api/v1/heartbeat` | Trạng thái thiết bị |
| `GET /api/v1/config?since=<version>` | Cấu hình mới và danh sách `parcel_id` được phép quét lại |
| `GET /api/v1/models/active` | Metadata model đang active và đường dẫn tải |
| `GET /api/v1/models/{version}/file` | File ONNX |

Mẫu `meta`:

```json
{
  "event_uuid": "7b0c2e1a-0d4f-4c1a-9a53-1e8d2f6b7c90",
  "ts_device": "2026-11-20T03:15:22.481Z",
  "parcel_id": "PKG000123",
  "qr_status": "OK",
  "warehouse": "A",
  "ai": {"probs": {"ok": 0.03, "rach": 0.91, "mop": 0.02, "vo": 0.03, "uot": 0.01}, "model": "v1-20261120"},
  "tray": "R",
  "reason": "DAMAGE_RACH",
  "alert": false,
  "latency_ms": 412
}
```

Bảo vệ API: giới hạn kích thước ảnh (1 MB), kiểm định dạng bằng magic bytes, kiểm schema bằng pydantic, giới hạn tốc độ theo thiết bị, ghi `ts_received` riêng (không tin hoàn toàn `ts_device`).

### 13.6 Các màn hình

| Màn hình | Nội dung |
| --- | --- |
| Đăng nhập | Tài khoản, mật khẩu, mã 2FA |
| Tổng quan | Thẻ KPI hôm nay (tổng, A, B, loại, tỉ lệ hư), biểu đồ theo giờ, 10 sự kiện gần nhất, cảnh báo bảo mật chưa xử lý |
| Kiện | Danh sách, tạo mới, nhập CSV, in nhãn, mở khóa quét lại |
| Sự kiện | Bảng có lọc, trang chi tiết với ảnh và xác suất, xuất CSV |
| Duyệt ảnh | Lưới ảnh, 5 nút nhãn kèm phím tắt, bỏ qua, xuất dataset |
| Thống kê | Biểu đồ theo ngày, kho, lớp lỗi, độ đúng của AI |
| Thiết bị | Trạng thái, phiên bản, đăng ký, thu hồi key |
| Model và Cấu hình | Upload và kích hoạt model, sửa ngưỡng và chính sách |
| Người dùng và Audit | Quản lý tài khoản, xem nhật ký |

### 13.7 Quy trình sử dụng điển hình

1. Operator tạo 20 kiện (kho A hoặc B), in nhãn PDF, dán lên hộp.
2. Đặt hộp lên băng chuyền, hệ thống phân loại.
3. Sự kiện hiện ra trên ThingsBoard và Console.
4. Hộp bị loại vì hư hại: người duyệt vào Duyệt ảnh, xác nhận hoặc sửa nhãn.
5. Mỗi tuần Admin xuất gói feedback, upload lên Drive `data/feedback/`, gán nhãn lại, tạo dataset phiên bản mới, train lại theo mục 9.
6. Admin upload model mới vào Console, đặt active, các Pi tự cập nhật.

### 13.8 Bảo mật Console

- HTTPS bắt buộc, mật khẩu tối thiểu 12 ký tự, khóa 15 phút sau 5 lần sai, 2FA bắt buộc cho Admin.
- Cookie `HttpOnly`, `Secure`, `SameSite=Strict`, thêm CSRF token cho form.
- Kiểm quyền ở server cho mọi route, không chỉ ẩn nút trên giao diện.
- Upload kiểm kích thước và định dạng, tên file do server tự đặt.
- Bí mật (HMAC key, JWT secret, API key) để trong biến môi trường hoặc file `.env`, nằm ngoài git.
- Sao lưu SQLite hằng ngày (`sqlite3 .backup`) cùng thư mục ảnh.

### 13.9 Thứ tự build Console

| Mốc | Nội dung |
| --- | --- |
| C1 | Khung FastAPI, DB, đăng nhập, phân quyền |
| C2 | API `events` và `heartbeat`, màn hình Sự kiện |
| C3 | Quản lý kiện, sinh QR, PDF nhãn (làm sớm vì cần cho test QR đầu cuối) |
| C4 | Tổng quan và Thống kê |
| C5 | Duyệt ảnh và xuất dataset |
| C6 | Thiết bị, Model, Cấu hình |
| C7 | Audit, 2FA, rà soát bảo mật |

Mỗi mốc kèm test bằng `pytest` và `httpx` (TestClient) cho API.

## 14. ThingsBoard

### 14.1 Dựng ThingsBoard CE

- Chạy bằng Docker (`thingsboard/tb-postgres`) trong `deploy/docker-compose.yml`, giao diện cổng 8080.
- Bật MQTT qua TLS (cổng 8883) và xác thực thiết bị bằng chứng chỉ X.509. Tên biến cấu hình và cách tạo keystore phụ thuộc phiên bản, làm theo tài liệu chính thức mục MQTT over SSL và X.509 certificate authentication của đúng phiên bản bạn dùng.
- Ở giai đoạn laptop có thể dùng access token và cổng 1883 không mã hóa để chạy nhanh, nhưng **trước khi demo bảo mật phải chuyển sang TLS + X.509**.

### 14.2 Thiết bị và dữ liệu gửi lên

Tạo device profile `ParcelSorterPi` và thiết bị `sorter-pi-01`. Pi gửi sự kiện lên topic `v1/devices/me/telemetry`:

```json
{
  "total": 128, "count_a": 51, "count_b": 49, "count_reject": 28,
  "count_rach": 9, "count_mop": 7, "count_vo": 4, "count_uot": 2,
  "count_uncertain": 3, "qr_fail": 3, "security_alerts": 1,
  "last_reason": "DAMAGE_RACH", "last_tray": "R",
  "last_conf": 0.91, "last_latency_ms": 412
}
```

- Bộ đếm là giá trị tích lũy lưu trong SQLite trên Pi, nên Pi khởi động lại vẫn không mất.
- Thuộc tính (attributes) thiết bị: `app_version`, `model_version`, `fw_version`, `cpu_temp`, `throttled`, `uart_ok`, `outbox_size`.
- Ảnh hàng lỗi: gửi thumbnail 160 x 160, JPEG chất lượng 60 (khoảng vài KB) dạng base64 ở khóa `last_thumb` để widget hiển thị. Kiểm tra giới hạn kích thước chuỗi của phiên bản ThingsBoard bạn dùng, nếu vướng thì chỉ hiển thị link mở sự kiện trong Console.

### 14.3 Dashboard

| Widget | Dữ liệu |
| --- | --- |
| Thẻ số lớn | `total`, `count_a`, `count_b`, `count_reject` |
| Biểu đồ tròn | Phân bố lý do loại: rach, mop, vo, uot, uncertain, QR |
| Biểu đồ đường | Số kiện theo thời gian |
| Ảnh hàng lỗi | `last_thumb` kèm `last_reason`, `last_conf` |
| Bảng sự kiện gần nhất | Timeseries table các khóa `last_*` |
| Trạng thái thiết bị | Thuộc tính `active`, `cpu_temp`, `uart_ok` |
| Bảng cảnh báo | Alarm (xem dưới) |
| Nút điều khiển (chỉ Admin) | Start, Stop, Emergency Stop, Reset counters |

Quy tắc cảnh báo (Alarm rules): `security_alerts` tăng, `uart_ok = false`, `cpu_temp` trên 80 độ C, thiết bị offline quá 60 s, `outbox_size` vượt 500.

### 14.4 Điều khiển từ xa (RPC) có kiểm soát

- Pi lắng nghe `v1/devices/me/rpc/request/+`.
- Whitelist method: `ctl_start`, `ctl_stop`, `ctl_estop`, `reset_counters`. Method khác: trả lỗi, ghi log, tăng `security_alerts`.
- Lệnh nguy hiểm (kích hoạt OTA) không đi qua RPC mà phải qua Console và có Admin xác nhận (mục 16).
- Mọi RPC được ghi log kèm thời điểm và tên tài khoản (ThingsBoard cung cấp audit log, Pi ghi thêm log cục bộ).

### 14.5 Tài khoản và phân quyền

- **Admin**: tài khoản Tenant Administrator, có dashboard điều khiển với nút RPC.
- **Viewer**: tài khoản Customer user, chỉ được gán dashboard bản chỉ xem (không có widget RPC).
- Bật 2FA (TOTP) cho tài khoản Admin trong phần Security của hồ sơ tài khoản ThingsBoard.
- Demo bảo mật: đăng nhập Viewer thử gửi RPC, kết quả phải bị từ chối.

## 15. Bảo mật

### 15.1 Mô hình mối đe dọa và biện pháp

| Mối đe dọa | Biện pháp | Cách demo |
| --- | --- | --- |
| QR giả hoặc bị sửa | HMAC-SHA256 trong QR (mục 6) | In QR sửa 1 ký tự, hệ thống báo QR\_INVALID |
| Chụp lại QR cũ để dùng lại (replay) | Bảng `seen_parcels`, kiểm hạn | Quét lại cùng mã, báo QR\_REPLAY |
| Nghe lén, giả mạo dữ liệu Pi gửi lên | MQTT qua TLS cổng 8883 với chứng chỉ X.509 hai chiều | Bắt gói bằng Wireshark chỉ thấy TLS; kết nối không có chứng chỉ bị từ chối |
| Truy cập dashboard trái phép | Phân quyền Admin và Viewer, 2FA | Viewer không có nút điều khiển, RPC từ Viewer bị từ chối |
| Gửi lệnh điều khiển lạ | RPC whitelist, UART whitelist, CRC, SEQ | Gửi method ngoài danh sách, bị từ chối và ghi log |
| Chiếm quyền Raspberry Pi | Hardening (SSH khóa, tường lửa, user riêng, file khóa quyền 600) | Thử đăng nhập SSH bằng mật khẩu bị từ chối, `ufw status` |
| Thay model độc hại | Kiểm `sha256` (và chữ ký) trước khi nạp | TC-17 |
| Firmware độc hại trên ESP32 | Secure OTA (ký ECDSA, kiểm hash, chống hạ phiên bản), Secure Boot v2, Flash Encryption | Mục 16 |
| Tấn công Console (brute force, CSRF, SQL injection, upload) | Khóa tài khoản, CSRF token, ORM, kiểm upload | Thử nhập sai 5 lần bị khóa |
| Mất mạng, mất liên lạc | Offline-first, watchdog | TC-10, TC-11 |

### 15.2 Cách làm từng hạng mục

**1. Xác thực thiết bị bằng X.509 (mutual TLS).** Tạo CA nội bộ bằng OpenSSL, cấp chứng chỉ cho server ThingsBoard và cho từng thiết bị.

```
openssl genrsa -out ca.key 4096
openssl req -x509 -new -key ca.key -sha256 -days 1825 -subj '/CN=Sorter Lab CA' -out ca.pem

openssl genrsa -out device.key 2048
openssl req -new -key device.key -subj '/CN=sorter-pi-01' -out device.csr
openssl x509 -req -in device.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out device.pem -days 365 -sha256
```

Chứng chỉ server cần thêm trường `subjectAltName` trùng tên máy chủ mà Pi dùng để kết nối. Khóa `ca.key` cất offline (USB), `device.key` trên Pi có quyền 600. Trong ThingsBoard, đặt credentials của thiết bị là loại X.509 và dán nội dung `device.pem`. Phía Pi: `client.tls_set(ca_certs='ca.pem', certfile='device.pem', keyfile='device.key')`, cổng 8883.

**2. Xác thực người dùng dashboard.** Phân quyền Admin và Viewer, bật 2FA cho Admin (mục 14.5).

**3. Xác thực mã QR.** Mục 6.

**4. Kênh truyền.** MQTT qua TLS (mục 14), Console qua HTTPS.

**5. RPC có kiểm soát.** Whitelist method, ghi log (mục 14.4). Không cho phép thêm nút "gõ lệnh tùy ý", đúng như đề cũ yêu cầu.

**6. UART có CRC và whitelist.** Mục 5.

**7. Hardening Raspberry Pi.** Thứ tự thực hiện, làm xong từng bước và kiểm tra còn đăng nhập được rồi mới sang bước sau:

- Cập nhật hệ thống: `sudo apt update && sudo apt full-upgrade -y`.
- Tạo user riêng `sorter` (không có sudo) để chạy dịch vụ. Tài khoản quản trị chỉ đăng nhập bằng khóa SSH.
- Copy khóa công khai bằng `ssh-copy-id`, **kiểm tra đăng nhập bằng khóa thành công** rồi mới đặt `PasswordAuthentication no` và `PermitRootLogin no` trong `/etc/ssh/sshd_config`, khởi động lại `ssh`.
- Tường lửa: `sudo ufw default deny incoming`, `sudo ufw allow from <dải LAN> to any port 22 proto tcp`, rồi mới `sudo ufw enable`. Pi chỉ cần kết nối ra ngoài tới ThingsBoard (8883) và Console (8443).
- Thư mục `/etc/sorter/` quyền 700, các file khóa quyền 600, chủ sở hữu là `sorter`.
- Tắt các dịch vụ không dùng (Bluetooth, avahi nếu không cần), bật cập nhật bảo mật tự động (`unattended-upgrades`).

**8. Bảo vệ ESP32: Secure Boot v2 và Flash Encryption.** Lưu ý trước khi làm:

- Hai tính năng này ghi eFuse, **không hoàn tác được**. Làm trên một board ESP32 thử riêng, không làm trên board chính cho đến khi chắc chắn.
- Secure Boot v2 trên ESP32 gốc chỉ có ở bản chip revision v3.0 trở lên. Kiểm tra bằng `esptool chip_id` (hoặc `espefuse summary`) trước khi mua và trước khi làm.
- ESP32 gốc dùng chữ ký RSA-3072 cho Secure Boot v2, khác với ECDSA P-256 của manifest OTA ở mục 16. Hai lớp này độc lập và bổ sung cho nhau.
- Bắt đầu bằng chế độ Development để thử, chỉ chuyển sang Release khi đã ổn định. Chế độ Release khóa chặt cổng nạp bằng UART và JTAG.
- **Sao lưu khóa ký Secure Boot ở nơi an toàn.** Mất khóa là không thể cập nhật firmware nữa.
- Dùng framework ESP-IDF (qua PlatformIO) cho phần này, làm theo đúng tài liệu của phiên bản ESP-IDF đang dùng.
- Đây là hạng mục Could trong bảng ưu tiên. Nếu thiếu thời gian thì trình bày trong báo cáo như hướng phát triển và chỉ demo Secure OTA.

**9. Chống phát lại (replay).** QR dùng bảng `seen_parcels`; UART dùng SEQ tăng dần; OTA dùng kiểm tra phiên bản tăng dần.

### 15.3 Quản lý khóa bí mật

| Khóa | Lưu ở đâu | Quy tắc |
| --- | --- | --- |
| HMAC của QR | `/etc/sorter/hmac.key` trên Pi (quyền 600) và biến môi trường của Console | Không đưa vào git. Đổi khóa thì phải in lại QR |
| Khóa riêng của CA | USB cất offline | Chỉ cắm khi cấp chứng chỉ |
| Khóa riêng của thiết bị | Trên đúng thiết bị đó, quyền 600 | Chứng chỉ thiết bị hết hạn sau 365 ngày, có lịch gia hạn |
| Khóa ký OTA (ECDSA) | Chỉ trên máy dev, mã hóa bằng passphrase | Không bao giờ upload hay commit |
| Khóa Secure Boot | Sao lưu offline | Không mất |
| API key thiết bị của Console | Chỉ lưu bản băm trong DB | Hiện một lần khi tạo, thu hồi được |

Dùng công cụ quét bí mật (ví dụ gitleaks) làm pre-commit để chặn việc vô tình đẩy khóa lên git.

### 15.4 Danh sách kiểm thử bảo mật

| Mã | Kiểm tra | Kỳ vọng |
| --- | --- | --- |
| S-01 | Bắt gói MQTT bằng Wireshark | Chỉ thấy TLS, không thấy nội dung rõ |
| S-02 | Kết nối MQTT 8883 không có chứng chỉ thiết bị | Bị từ chối |
| S-03 | SSH vào Pi bằng mật khẩu | Bị từ chối |
| S-04 | `sudo ufw status` | Mặc định chặn đến, chỉ mở SSH từ LAN |
| S-05 | Viewer gửi RPC | Bị từ chối |
| S-06 | Đăng nhập Console sai 5 lần | Tài khoản bị khóa tạm |
| S-07 | TC-05, TC-06, TC-07, TC-12, TC-13, TC-16, TC-17 | Như mục 11.5 |
| S-08 | Các kịch bản OTA (a) đến (e) ở mục 16.5 | Như mục 16.5 |

## 16. App nạp firmware bảo mật (Secure OTA)

### 16.1 Luồng tổng thể

1. Dev build `firmware.bin` bằng PlatformIO.
2. **OTA Tool** (chạy trên máy dev, nơi giữ khóa riêng): chọn file, tính SHA-256, ký bằng ECDSA P-256, tạo `manifest.json`, upload bản đã ký lên Console.
3. **Admin** vào Console (đã qua 2FA), chọn bản firmware và thiết bị, bấm "Phát hành OTA" và xác nhận. Console ghi vào audit log và đặt lệnh chờ cho thiết bị.
4. Pi nhận lệnh qua phản hồi heartbeat (trường `pending_ota` gồm `version`, `manifest_url`), kiểm tra thiết bị đang READY, rồi chuyển cho ESP32 bằng khung UART `OTA,<manifest_url>`. **Pi chỉ làm trạm chuyển tiếp, ESP32 không tin Pi**, tự kiểm lại mọi thứ.
5. **ESP32**: tải manifest qua HTTPS, kiểm chữ ký bằng khóa công khai nhúng sẵn trong firmware, kiểm phiên bản mới lớn hơn phiên bản hiện tại, tải file vào phân vùng OTA đang rảnh, tính SHA-256 trong lúc tải và so với manifest, đặt phân vùng khởi động mới, khởi động lại.
6. Sau khi khởi động, firmware mới tự kiểm tra (UART nói chuyện được, đọc được cảm biến) trong 30 giây rồi mới gọi `esp_ota_mark_app_valid_cancel_rollback()`. Nếu không xác nhận kịp, bootloader tự quay về bản cũ.
7. ESP32 báo `EVT,seq,OTA_OK` hoặc `OTA_FAIL`, Console ghi lịch sử.

### 16.2 Khóa và định dạng manifest

```
openssl ecparam -name prime256v1 -genkey -noout -out ota_ec.pem
openssl pkcs8 -topk8 -v2 aes-256-cbc -in ota_ec.pem -out ota_priv.pem     # nhap passphrase
openssl ec -in ota_ec.pem -pubout -out ota_pub.pem                        # nhung vao firmware
# xoa an toan ota_ec.pem (ban khong ma hoa) sau khi tao xong
```

```json
{
  "device": "esp32-sorter",
  "version": "1.2.0",
  "url": "https://console.local:8443/fw/esp32-sorter-1.2.0.bin",
  "size": 842160,
  "sha256": "<hex>",
  "created_at": "2026-12-01T09:00:00Z",
  "signature": "<base64 chu ky ECDSA DER>"
}
```

Chuỗi được ký là `device|version|size|sha256|created_at` (cố định, không phụ thuộc cách sắp xếp JSON). `url` không cần ký vì toàn vẹn nội dung đã được đảm bảo bởi `sha256` nằm trong phần ký.

```python
import hashlib, base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

def sign_manifest(priv_pem: bytes, passphrase: bytes, bin_path, device, version, created_at, url):
    data = open(bin_path, 'rb').read()
    sha = hashlib.sha256(data).hexdigest()
    msg = f'{device}|{version}|{len(data)}|{sha}|{created_at}'.encode()
    key = serialization.load_pem_private_key(priv_pem, passphrase)
    sig = key.sign(msg, ec.ECDSA(hashes.SHA256()))        # DER
    return {'device': device, 'version': version, 'url': url, 'size': len(data),
            'sha256': sha, 'created_at': created_at,
            'signature': base64.b64encode(sig).decode()}
```

Phía ESP32 dùng mbedTLS: nạp khóa công khai bằng `mbedtls_pk_parse_public_key`, băm chuỗi ký bằng SHA-256 rồi kiểm bằng `mbedtls_pk_verify` với chữ ký DER.

### 16.3 Đặc tả OTA Tool (ứng dụng desktop)

Python với Tkinter hoặc PySide6, chạy trên máy dev.

| Chức năng | Yêu cầu |
| --- | --- |
| Mở khóa | Nhập passphrase để mở khóa riêng, tự khóa lại sau 5 phút không dùng |
| Chọn firmware | Chọn file `.bin`, nhập phiên bản dạng semver. Tool từ chối nếu phiên bản không lớn hơn bản đã phát hành gần nhất |
| Hiển thị | Kích thước và SHA-256 của file |
| Ký và tạo manifest | Sinh `manifest.json` như mục 16.2 |
| Phát hành | Upload `.bin` và manifest lên Console (API dành cho Admin) hoặc lưu ra thư mục để copy tay |
| Lịch sử | Bảng các lần phát hành: thời gian, phiên bản, SHA-256, người ký, kết quả trên từng thiết bị (lấy từ Console) |
| Chế độ demo tấn công | Tạo bản firmware bị sửa 1 byte (giữ nguyên manifest cũ) và manifest ký bằng khóa khác, để chứng minh ESP32 từ chối |

Khóa riêng không bao giờ rời máy dev, không đưa vào git, không gửi qua mạng.

### 16.4 Phía ESP32: yêu cầu

- Bảng phân vùng có `ota_0`, `ota_1` và `otadata`; bật tính năng app rollback trong cấu hình bootloader.
- Tải qua HTTPS, kiểm chứng chỉ máy chủ bằng CA nhúng sẵn (hoặc ghim chứng chỉ).
- Chỉ nhận OTA khi băng chuyền ở trạng thái READY. Từ chối khi đang chạy hoặc đang chờ SORT.
- Có timeout cho toàn bộ quá trình, thất bại thì giữ nguyên bản cũ và phát `OTA_FAIL` kèm mã lỗi.
- Lưu số phiên bản đã chạy vào NVS để chống hạ phiên bản.

### 16.5 Kịch bản kiểm thử và demo OTA

| Mã | Kịch bản | Kỳ vọng |
| --- | --- | --- |
| (a) | Firmware hợp lệ, phiên bản mới | Nạp thành công, `fw_version` trên dashboard tăng |
| (b) | Sửa 1 byte trong `.bin` sau khi ký | SHA-256 không khớp, từ chối, giữ bản cũ |
| (c) | Manifest ký bằng khóa khác | Chữ ký sai, từ chối |
| (d) | Firmware ký hợp lệ nhưng cố ý lỗi khi khởi động | Không xác nhận kịp, tự quay về bản cũ |
| (e) | Firmware hợp lệ nhưng phiên bản thấp hơn hiện tại | Từ chối (chống hạ phiên bản) |

## 17. Kịch bản demo

| Bước | Thao tác | Kết quả cần thấy |
| --- | --- | --- |
| 1 | Hộp tốt, QR kho A | Vào khay A, dashboard tăng A |
| 2 | Hộp tốt, QR kho B | Vào khay B |
| 3 | Hộp rách | Khay Loại, dashboard hiện ảnh và `rach` kèm độ tin cậy |
| 4 | Hộp móp | Khay Loại, mã DAMAGE\_MOP |
| 5 | Hộp vỡ hoặc thủng | Khay Loại, mã DAMAGE\_VO |
| 6 | Hộp ướt | Khay Loại, mã DAMAGE\_UOT |
| 7 | QR giả (in bản sửa 1 ký tự) | Khay Loại, đèn đỏ, còi, cảnh báo bảo mật |
| 8 | Quét lại QR đã dùng | QR\_REPLAY |
| 9 | Admin bấm dừng khẩn cấp trên dashboard | Băng chuyền dừng. Đăng nhập Viewer thì không có nút |
| 10 | Rút dây UART hoặc ngắt mạng | Watchdog dừng băng, hoặc hệ thống vẫn chạy rồi đồng bộ bù khi có mạng |
| 11 | Secure OTA: bản hợp lệ rồi bản sửa 1 byte | Bản đầu thành công, bản sau bị từ chối |
| 12 | Console: duyệt ảnh lỗi, xuất dataset | Chứng minh vòng cải thiện model |
| 13 | Slide kết quả | Accuracy, confusion matrix, độ trễ suy luận trên Pi, tóm tắt bảo mật |

Mẹo chuẩn bị: đánh số sẵn 12 đến 15 hộp theo đúng thứ tự demo, in sẵn QR (cả QR giả), quay video dự phòng, chạy thử toàn bộ ít nhất 3 lần trong đúng điều kiện ánh sáng của phòng demo, mang theo nguồn và dây dự phòng.

## 18. Lộ trình, phân công, rủi ro

### 18.1 Lộ trình đề xuất (10 tuần, điều chỉnh theo thời gian thật của bạn)

| Tuần | Công việc chính |
| --- | --- |
| 1 | Chốt spec, dựng repo, mua linh kiện, thu gom hộp (từ 80 hộp), chụp dataset v0, bước L0 |
| 2 | L2 (QR và HMAC), L4 (decision), dựng khung giàn băng chuyền |
| 3 | Train v0 trên Colab, L3 (benchmark laptop), L5 (main và mock), firmware ESP32 cơ bản (motor, cảm biến, UART) |
| 4 | L6 (uplink, Console C1 đến C3, ThingsBoard Docker), L7 (ESP32 qua USB), **qua cổng chuyển sang Pi**. Thử sớm TLS và X.509 trên ThingsBoard |
| 5 | Hoàn thiện giàn: đèn, camera, che sáng. Chụp dataset v1 (ít nhất 3 phiên). P1 đến P6 |
| 6 | Train v1, so sánh E1 đến E5, chọn ngưỡng, benchmark trên Pi, hiệu chỉnh cơ khí (P7) |
| 7 | Chạy TC-01 đến TC-18 trên Pi (P8) và sửa lỗi. Console C4, C5 |
| 8 | Bảo mật: TLS, X.509, hardening, 2FA. Console C6, C7 |
| 9 | Secure OTA (tool và ESP32). Secure Boot trên board thử nếu còn thời gian. Soak test |
| 10 | Đóng băng tính năng, báo cáo, slide, tập demo, quay video dự phòng |

Quan trọng: giàn băng chuyền với camera và đèn cố định phải xong trước tuần 5 để chụp dataset v1. Nếu giàn bị trễ, dựng một khung tạm giữ đúng chiều cao camera, đèn và nền để chụp, đừng chờ.

### 18.2 Phân công gợi ý (nhóm 3 đến 4 người)

| Vai trò | Phụ trách |
| --- | --- |
| AI và dữ liệu | Mục 7 đến 10: thu thập, gán nhãn, train, đánh giá, export, benchmark |
| Edge Pi | Mục 11 đến 12: camera, QR, AI inference, quyết định, uplink |
| ESP32 và cơ khí | Mục 3, 5, 12.7, phần ESP32 của mục 16: giàn băng chuyền, firmware |
| Console, ThingsBoard, OTA Tool | Mục 13, 14, 16: phần mềm quản lý, dashboard, công cụ nạp firmware |

Nhóm ít người hơn thì cắt hạng mục Could trước, sau đó M7, M8, M10 của Console.

### 18.3 Rủi ro và cách giảm thiểu

| Rủi ro | Cách giảm thiểu |
| --- | --- |
| Dataset ít hoặc lệch | Thu nhiều hộp khác nhau, chia theo hộp, thêm phiên chụp. Nếu confusion matrix cho thấy `rach` và `vo` nhầm nhau nhiều, gộp thành một lớp "rách-thủng" |
| Chạy tốt trên laptop, kém trên Pi | Chụp dataset trên chính giàn thật, khóa phơi sáng, che sáng, so sánh ảnh đầu vào giữa hai nơi |
| Pi quá chậm hoặc bị throttle | Quạt tản nhiệt, giảm số khung từ 3 xuống 1, INT8, giảm kích thước ảnh xuống 224, đổi sang MobileNetV3-Small |
| Servo gạt không chuẩn | Thêm cảm biến thứ 2 tại vị trí gạt, giảm tốc băng chuyền, hiệu chỉnh `T_TRAVEL_MS` |
| Secure Boot hoặc Flash Encryption làm hỏng board ESP32 | Làm trên board thử riêng, bắt đầu bằng Development mode, sao lưu khóa |
| Colab ngắt phiên | Checkpoint ra Drive, dùng `resume=True` |
| TLS và X.509 trên ThingsBoard khó cấu hình | Làm thử từ tuần 4. Phương án dự phòng: TLS phía server với access token |
| Nhiễu UART hoặc sụt nguồn | Tách nguồn, GND chung, tụ lọc, dây ngắn |
| Phạm vi quá rộng | Bám bảng Must, Should, Could ở mục 1.1, cắt Could trước |
| Lộ khóa trong git | `.gitignore`, quét bí mật khi commit |

## Phụ lục A. Tiêu chí hoàn thành (Definition of Done)

**Giai đoạn L (laptop):**

- Toàn bộ unit test xanh, gồm đủ 10 dòng của bảng quyết định.
- 50 ảnh phát lại cho kết quả khay và mã lý do đúng 100% so với kỳ vọng ghi sẵn.
- Có bảng benchmark trên laptop.
- Sự kiện hiện đúng trên ThingsBoard và Console, outbox hoạt động khi mất mạng.
- Firmware ESP32 chạy đúng PING/PONG, CRC, SORT, watchdog khi nối laptop qua USB.

**Giai đoạn P (Pi):**

- Có bảng benchmark trên Pi, đã chọn model cuối.
- Dataset v1 chụp trên giàn thật, đã chia theo hộp, test có phiên riêng.
- TC-01 đến TC-18 đạt trên giàn thật.
- Đạt KPI ở mục 1.3 trong lần soak test 30 phút hoặc 100 kiện.
- Dịch vụ systemd tự khởi động và tự phục hồi.

**Bảo mật và demo:**

- S-01 đến S-08 đạt.
- Kịch bản demo mục 17 chạy trơn tru 3 lần liên tiếp.
- Báo cáo có trích dẫn dataset ngoài và nêu rõ giới hạn đã biết (mục 1.2).

## Phụ lục B. Tài liệu chính thức nên đọc

Vì các cấu hình dưới đây thay đổi theo phiên bản, hãy đọc tài liệu đúng phiên bản bạn cài thay vì làm theo trí nhớ:

- ESP-IDF (docs.espressif.com): Secure Boot v2, Flash Encryption, OTA, App Rollback.
- ThingsBoard (thingsboard.io/docs): MQTT over SSL, X.509 certificate authentication, Two-factor authentication, RPC, Alarms.
- Ultralytics (docs.ultralytics.com): Image Classification, Export (ONNX), tham số huấn luyện.
- ONNX Runtime (onnxruntime.ai/docs): Quantization, hiệu năng trên ARM.
- Picamera2 manual (có trên trang tài liệu của Raspberry Pi): điều khiển phơi sáng, lấy nét.
- Roboflow Universe và Kaggle: dataset công khai, luôn kiểm tra giấy phép từng bộ.
