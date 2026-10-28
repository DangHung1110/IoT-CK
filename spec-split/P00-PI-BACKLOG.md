# P00 - Backlog chuyển sang Raspberry Pi

Trạng thái: BACKLOG, không thực thi trong phạm vi hiện tại.

## Entry criteria

- L06 đạt.
- Có Raspberry Pi 4 64-bit, nguồn/tản nhiệt và camera đã chọn.
- Có dataset v1 hoặc kế hoạch chụp bằng đúng giàn/camera/đèn.

## Phases tương lai

1. `P01 OS/runtime`: Raspberry Pi OS 64-bit, Python 3.11, service user, systemd.
2. `P02 camera`: Picamera2 adapter, manual exposure/white balance/focus, ROI parity.
3. `P03 model`: CPU benchmark 1/2/4 threads, nhiệt/throttle, chọn FP32 hoặc INT8.
4. `P04 serial`: SerialController với ESP32 thật, retry/CRC/SEQ/watchdog.
5. `P05 calibration`: timing, servo/sensor/motor; đây mới là phần lắp ráp.
6. `P06 network`: MQTT TLS/ThingsBoard/Console HTTPS.
7. `P07 security`: hardening Pi, key permissions, audit.
8. `P08 acceptance`: TC-01..TC-18, soak 100 kiện/30 phút.

## Pi-specific gate

- Cùng ONNX/card cho decision parity với laptop.
- Preprocess output hash gần/đúng với golden fixtures.
- p95 inference <= 250 ms hoặc có ADR thay model/input/threads.
- Không throttle trong soak test.
- Service restart không làm duplicate/replay sai.

## Không làm sớm

- Không lượng tử hóa chỉ vì “có thể”; chỉ giữ INT8 nếu latency tốt hơn và recall giảm trong giới hạn đã chấp nhận.
- Không bật Secure Boot/Flash Encryption ESP32 trên board chính ở phase Pi cơ bản.
