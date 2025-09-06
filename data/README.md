# Mô tả dữ liệu mực nước sông Hồng
Dữ liệu thực tế được thu thập trên 3 trạm thuỷ văn Hà Nội, Vụ Quang và Hưng Yên. Mỗi tệp chứa thông tin mực nước tại từng khung giờ cách đều trong các ngày liên tiếp.

Dữ liệu này giúp theo dõi biến động mực nước, phục vụ cho việc phân tích thủy văn và dự báo lũ lụt tại khu vực sông Hồng.

<table>
  <tr>
    <th>STT</th>
    <th>Dataset</th>
    <th>Interval</th>
    <th>NumSamples</th>
    <th>Frequency</th>
    <th>Max</th>
    <th>Min</th>
    <th>Medium</th>
  </tr>
  <tr>
    <td>1</td>
    <td>Vu Quang</td>
    <td>2008-2017</td>
    <td>29224</td>
    <td>3 hours</td>
    <td>1762.5</td>
    <td>480</td>
    <td>830.70</td>
  </tr>
    <tr>
    <td>2</td>
    <td>Ha Noi</td>
    <td>2008-2017</td>
    <td>29224</td>
    <td>3 hours</td>
    <td>1042</td>
    <td>10</td>
    <td>254.85</td>
  </tr>
    <tr>
    <td>3</td>
    <td>Hung Yen</td>
    <td>2008-2015</td>
    <td></td>
    <td>1 hour</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

# Phương pháp tạo dữ liệu huấn luyện

## EDA
- Thông qua quá trình phân tích dữ liệu (EDA), các đồ thị ACF và PACF cho ta thấy mức độ ảnh hưởng truy hồi giữa các bước thời gian với nhau (p-values & q-values).

## Sliding windows
- Dựa vào chỉ số p-values, ta quyết định tạo các cửa số cắt với bước thời gian quá khứ tương ứng là p bước.
- Dựa vào từng phương pháp dự báo (multi-step ahead or multi-step autoregresive) và số bước dự báo mong muốn, số bước thời gian làm nhãn dự báo là tương ứng.
