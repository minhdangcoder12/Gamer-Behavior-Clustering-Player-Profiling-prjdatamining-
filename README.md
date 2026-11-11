# Gamer Behavior Clustering & Player Profiling

Dự án khai phá dữ liệu phân khúc người chơi game online theo lối chơi. Dự án kết hợp **Agglomerative Hierarchical Clustering** với **K-Means**, đánh giá chất lượng cụm bằng **Silhouette Score** và **Davies-Bouldin Index**, sau đó mô tả hồ sơ từng phân khúc và đề xuất chiến lược monetization.

> **Trạng thái:** đã hoàn thiện toàn bộ pipeline trong một notebook: EDA, tạo đặc trưng, chọn k, phân cụm Hybrid, đánh giá, profiling và chiến lược. Notebook chạy lại thành công từ đầu đến cuối ngày 11/11/2026, không có cell lỗi, xuất 9 biểu đồ và 2 bảng kết quả vào `reports/figures/`.

## Mục tiêu

- Trích xuất chỉ số lối chơi: thời gian chơi, kỹ năng, mức độ sưu tập.
- Phân cụm người chơi bằng **Agglomerative (Ward)** kết hợp **K-Means**.
- Đánh giá bằng **Silhouette Score** và **Davies-Bouldin Index**, so sánh với từng thuật toán đơn lẻ.
- Mô tả hồ sơ từng phân khúc (Casual, Hardcore, …) và đề xuất chiến lược monetization.

Phạm vi phân khúc dựa trên **hành vi chơi**. Tên phân khúc là diễn giải của nhóm dựa trên giá trị trung bình của cụm, không phải nhãn có sẵn trong dữ liệu.

## Dataset

Nguồn: [Kaggle — Predict Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset), tác giả Rabie El Kharoua.

File: `data/online_gaming_behavior_dataset.csv`. File không được commit; notebook tự tải qua `kagglehub` nếu chưa có.

| Thuộc tính | Giá trị |
| --- | --- |
| Số dòng | 40.034 |
| Số cột | 13 |
| Giá trị thiếu | 0 |
| Dòng trùng lặp | 0 |
| Có mua hàng trong game (`InGamePurchases = 1`) | ≈ 20% |
| EngagementLevel Low / Medium / High | 26% / 48% / 26% |

Các biến phân loại chia gần như đều tuyệt đối (5 thể loại game, mỗi loại ≈ 20%; độ khó đúng 50/30/20) và các biến số có phân phối gần đều. Đây là dấu hiệu **dữ liệu tổng hợp**, nên ranh giới cụm được kỳ vọng không rõ ràng.

## Đặc trưng và quy tắc xử lý

| Chỉ số đề bài | Đặc trưng sử dụng | Công thức / ghi chú |
| --- | --- | --- |
| Playtime | `WeeklyPlayHours` | `SessionsPerWeek × AvgSessionDurationMinutes / 60` |
| Skill score | `PlayerLevel` | Proxy; dataset không có điểm kỹ năng |
| Item collection rate | `AchievementsUnlocked` | Proxy; dataset không có dữ liệu vật phẩm |
| Win/loss ratio | — | Không có trong dataset; ghi nhận là hạn chế |

Quy tắc:

- **Không dùng `PlayTimeHours`**: tương quan với `EngagementLevel` ≈ 0,00, trong khi `WeeklyPlayHours` đạt 0,77.
- **Không đưa `InGamePurchases` vào phân cụm**: biến nhị phân chi phối khoảng cách Euclid, khiến k = 2 chỉ tách "mua / không mua". Biến này chỉ dùng để mô tả hồ sơ.
- **Không dùng biến nhân khẩu học** (tuổi, giới tính, khu vực) khi phân cụm; chỉ dùng để kiểm tra sự khác biệt giữa các phân khúc.
- Chuẩn hóa ba đặc trưng bằng `StandardScaler`. Dữ liệu không thiếu và không có ngoại lai rõ, nên không cần điền giá trị thiếu hay chuẩn hóa robust.
- `SEED = 42` cố định cho lấy mẫu và K-Means để kết quả tái lập được.

Lý do chi tiết của từng quyết định: [SPEC.md](SPEC.md).

## Phương pháp

| Bước | Thuật toán | Vai trò |
| --- | --- | --- |
| 1 | Agglomerative (Ward) trên mẫu 5.000 người | Dendrogram + Silhouette/DB theo k = 2…8 để chọn k |
| 2 | Tâm cụm Agglomerative → `init` của K-Means | Khởi tạo xác định, có căn cứ |
| 3 | K-Means trên toàn bộ 40.034 người | Tinh chỉnh ranh giới cụm |
| Đối chứng | Agglomerative đơn thuần, K-Means++ (`n_init=10`) | So sánh với phương pháp Hybrid |

Agglomerative chỉ chạy trên mẫu vì ma trận khoảng cách của 40.034 điểm cần khoảng 6 GB RAM. Silhouette (độ phức tạp O(n²)) tính trên mẫu 10.000 người; Davies-Bouldin tính trên toàn bộ dữ liệu.

## Kết quả

**Chọn k = 4**: Silhouette gần như phẳng (0,21–0,24) với mọi k; Davies-Bouldin giảm mạnh tới k = 4 (1,64 → 1,20) rồi chững lại.

| Phương pháp (k = 4) | Silhouette ↑ | Davies-Bouldin ↓ |
| --- | --- | --- |
| Agglomerative (mẫu 5k) | 0,212 | 1,204 |
| K-Means++ (`n_init=10`) | 0,281 | 1,059 |
| **Hybrid Agglomerative → K-Means** | **0,276** | **1,059** |

Hybrid tốt hơn rõ Agglomerative đơn thuần và ngang K-Means++, nhưng chỉ cần một lần chạy và cho kết quả xác định. Adjusted Rand Index giữa Hybrid và K-Means++ chỉ 0,56: hai cách chia khác nhau dù điểm số gần bằng nhau, xác nhận dữ liệu không có ranh giới cụm tự nhiên rõ ràng.

| Phân khúc | Tỉ trọng | Giờ/tuần | Level | Thành tựu | Engagement High | Chiến lược gợi ý |
| --- | --- | --- | --- | --- | --- | --- |
| Hardcore | 19% | **36,1** | 50 | 23 | **92%** | Season pass, subscription, cosmetic cao cấp |
| Veteran Grinders | 26% | 10,2 | **77,7** | 33 | 13% | DLC, chế độ mới, prestige items |
| Achievement Hunters | 25% | 10,2 | 23,5 | **34,8** | 11% | Collection sets, XP booster |
| Casual | 30% | 9,6 | 47 | 9 | 8% | Starter pack, rewarded ads, ưu đãi lần đầu |

Tỉ lệ mua hàng ≈ 20% ở **mọi** phân khúc, nên dataset **không có nhóm High-Spender** tách biệt theo hành vi. `InGamePurchases` chỉ có 0/1, không có số tiền.

Báo cáo đầy đủ: [reports/REPORT.md](reports/REPORT.md).

## Công nghệ

Python 3.11, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, kagglehub và Jupyter Notebook.

## Cấu trúc dự án

```text
.
├── README.md
├── SPEC.md                         # Quyết định thiết kế: chỉ số thay thế, phương pháp
├── requirements.txt
├── .gitignore
├── data/
│   └── online_gaming_behavior_dataset.csv   # CSV nguồn, không commit
├── notebooks/
│   └── gamer_clustering.ipynb      # Notebook chính, đã có sẵn output
├── tools/
│   └── build_notebook.py           # Script sinh notebook
└── reports/
    ├── REPORT.md                   # Báo cáo kết quả
    └── figures/
        ├── 01_distributions.png        # Phân bố từng biến
        ├── 02_engagement_corr.png      # Tương quan với EngagementLevel
        ├── 03_feature_corr.png         # Tương quan giữa các đặc trưng
        ├── 04_dendrogram.png           # Cây phân cấp + đường cắt k = 4
        ├── 05_choose_k.png             # Silhouette / Davies-Bouldin theo k
        ├── 06a_method_comparison.png   # So sánh 3 phương pháp
        ├── 06_pca_clusters.png         # Các phân khúc trên PCA 2 chiều
        ├── 07_segment_profiles.png     # Hồ sơ 4 phân khúc
        ├── 08_purchase_rate.png        # Tỉ lệ mua hàng theo phân khúc
        ├── evaluation.csv              # Bảng chỉ số đánh giá
        └── segment_profile.csv         # Bảng hồ sơ phân khúc
```

Notebook được sinh từ `tools/build_notebook.py`. **Nếu sửa trực tiếp notebook thì không chạy lại script**, vì script sẽ ghi đè thay đổi.

## Cài đặt và chạy

### Cách 1: Google Colab (không cần cài đặt)

1. Vào [colab.research.google.com](https://colab.research.google.com), chọn **File → Upload notebook**, rồi chọn `notebooks/gamer_clustering.ipynb`.
2. **Runtime → Change runtime type → CPU** (scikit-learn không dùng GPU/TPU).
3. **Runtime → Run all**. Dataset tự tải qua `kagglehub`; biểu đồ được lưu vào thư mục `figures/` trên Colab.

Có thể dùng extension **Google Colab** cho VS Code: mở notebook, chọn **Select Kernel → Colab → New Colab Server**.

### Cách 2: Máy cá nhân

**1. Chuẩn bị:** Python 3.11 bản 64-bit, cùng VS Code với extension **Python** và **Jupyter** nếu muốn chạy notebook trong trình soạn thảo. Chạy các lệnh dưới đây tại thư mục gốc dự án.

**2. Tạo môi trường ảo và cài thư viện**

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Bash (Git Bash / Linux / macOS):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

| Thư viện | Công dụng |
| --- | --- |
| pandas, NumPy | Đọc dữ liệu, tạo đặc trưng |
| scikit-learn, SciPy | Chuẩn hóa, phân cụm, dendrogram, chỉ số đánh giá |
| Matplotlib, Seaborn | Trực quan hóa |
| kagglehub | Tải dataset từ Kaggle |
| Jupyter | Chạy notebook |

`pandas` được cố định ở **2.2.3**. Trên máy phát triển, các bản mới hơn bị **Smart App Control** của Windows chặn file DLL (`DLL load failed ... An Application Control policy has blocked this file`).

**3. Kiểm tra môi trường**

```powershell
.\.venv\Scripts\python.exe -c "import pandas, numpy, sklearn, scipy, matplotlib, seaborn, kagglehub; print('Môi trường OK')"
```

**4. Chạy notebook**

Chạy toàn bộ và lưu output vào notebook:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/gamer_clustering.ipynb
```

Hoặc trong VS Code: mở `notebooks/gamer_clustering.ipynb` → **Select Kernel → Python Environments → .venv** → **Run All**. Thời gian chạy khoảng 1–2 phút trên CPU. Biểu đồ và bảng kết quả được ghi vào `reports/figures/`.

Nếu không tải dataset tự động được, tải `online_gaming_behavior_dataset.csv` từ Kaggle và đặt vào `data/`.

## Hạn chế

- Không có win/loss ratio; skill score và item collection rate dùng proxy.
- `InGamePurchases` nhị phân, nên không đo được mức chi tiêu và không xác định được High-Spenders.
- Dữ liệu có dấu hiệu tổng hợp, nên Silhouette chỉ ở mức vừa phải (≈ 0,28); cần kiểm chứng trên dữ liệu thật.
- Agglomerative chạy trên một mẫu 5.000 người; chưa kiểm tra độ ổn định trên nhiều mẫu khác nhau.

## Thành viên

- Nguyễn Minh Đăng
- Phạm Hải Long

## Tài liệu tham khảo

- [scikit-learn — Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn — AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [scikit-learn — KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn — Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
- [SciPy — scipy.cluster.hierarchy](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
