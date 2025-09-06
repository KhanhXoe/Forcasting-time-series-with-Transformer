# Hướng dẫn cài đặt và chạy LSTM và Transformer Notebook cho dự báo mực nước

## 1. Tạo môi trường ảo (tuỳ chọn nhưng khuyến khích)
### Sử dụng venv:
```bash
python3 -m venv venv
source venv/bin/activate      # Trên Linux
venv\Scripts\activate       # Trên Windows
```

### Hoặc dùng conda:
```bash
conda create -n water_forecast python=3.10 -y
conda activate water_forecast
```

## 2. Cài đặt các thư viện cần thiết
Tạo file `requirements.txt` với nội dung sau:

```
torch==2.0.1
numpy==1.24.4
pandas==1.5.3
matplotlib==3.7.1
scipy==1.10.1
```

Sau đó chạy:
```bash
pip install -r requirements.txt
```

## 3. Mở và chạy file notebook
Cài notebook nếu chưa có:
```bash
pip install notebook
```

Sau đó chạy:
```bash
jupyter notebook
```
Và mở file `Transformers.ipynb`.

## 4. Hoặc chạy trên Google Colab
- Truy cập: https://colab.research.google.com
- Upload file notebook
- Thêm cell đầu tiên:
```python
!pip install torch==2.0.1 numpy==1.24.4 pandas==1.5.3 matplotlib==3.7.1 scipy==1.10.1
```
