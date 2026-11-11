# Spec — Gamer Behavior Clustering & Player Profiling

Ghi lại các quyết định thiết kế và lý do. Số liệu lấy từ lần chạy notebook ngày 11/11/2026.

## Mục tiêu
Phân khúc người chơi theo hành vi bằng Agglomerative Hierarchical + K-Means, đánh giá bằng Silhouette và Davies-Bouldin, đề xuất chiến lược monetization cho từng nhóm.

## Ánh xạ chỉ số đề bài → dataset
| Đề bài | Dataset | Ghi chú |
|---|---|---|
| playtime | `WeeklyPlayHours = SessionsPerWeek × AvgSessionDurationMinutes / 60` | `PlayTimeHours` không tương quan với EngagementLevel (r ≈ 0,00); WeeklyPlayHours r ≈ 0,77 |
| skill score | `PlayerLevel` | proxy, không có điểm kỹ năng thật |
| item collection rate | `AchievementsUnlocked` | proxy, không có dữ liệu vật phẩm |
| win/loss ratio | — | **không có trong dataset**, ghi rõ là hạn chế |
| chi tiêu | `InGamePurchases` (0/1) | dùng để profiling, **không** đưa vào phân cụm |

## Quyết định và căn cứ

| Quyết định | Căn cứ (số liệu) |
|---|---|
| Dùng 3 đặc trưng: giờ/tuần, level, thành tựu | Ablation: Silhouette 0,276 (tốt nhất); 5 biến gốc chỉ 0,144; thêm InGamePurchases giảm còn 0,250 |
| Không dùng `PlayTimeHours` | r ≈ 0,00 với engagement; thay nó vào bộ đặc trưng làm Silhouette giảm còn 0,268 |
| Không dùng `EngagementLevel` khi phân cụm | Gần như quy tắc ngưỡng theo giờ/tuần (Gradient Boosting đạt 93%, giờ/tuần chiếm gần hết importance); giữ lại để kiểm chứng |
| `StandardScaler` | Ward và K-Means dùng khoảng cách Euclid; dữ liệu không có ngoại lai |
| Agglomerative trên mẫu 5.000 | Ma trận khoảng cách 40k điểm cần ≈ 6 GB RAM |
| Linkage Ward | Single/Average tạo cụm lệch (cụm nhỏ nhất 1 và 79 người); Ward và Complete cân đối, cho kết quả Hybrid gần bằng nhau; Ward cùng hàm mục tiêu với K-Means |
| k = 4 | Silhouette phẳng (0,21–0,24); Davies-Bouldin giảm mạnh tới k = 4 (1,64 → 1,20) rồi chững lại |
| Silhouette trên mẫu 10.000, DB trên toàn bộ | Silhouette có độ phức tạp O(n²) |
| `SEED = 42` | Tái lập kết quả |

## Phương pháp
1. StandardScaler trên 3 đặc trưng.
2. Agglomerative (Ward) trên mẫu 5.000 dòng → dendrogram + metric theo k → chọn k.
3. Tâm cụm Agglomerative làm `init` cho K-Means trên toàn bộ 40.034 dòng.
4. So sánh: Agglomerative đơn thuần (mẫu), K-Means++ (`n_init=10`), Hybrid.
5. Kiểm tra: Hopkins, ablation, linkage, độ ổn định 10 lần chạy, χ², GMM, cây quyết định, Gradient Boosting.

## Kết quả (k = 4, Hybrid)
- Silhouette 0,276, Davies-Bouldin 1,059 (K-Means++: 0,281 / 1,059; Agglomerative: 0,212 / 1,204).
- Bốn nhóm: **Hardcore** (19%, ≈ 36 giờ/tuần), **Veteran Grinders** (26%, level ≈ 78), **Achievement Hunters** (25%, level ≈ 23, thành tựu ≈ 35), **Casual** (30%, thành tựu ≈ 9).
- Hopkins 0,58: xu hướng phân cụm yếu.
- Độ ổn định: Hardcore được tìm lại ở mọi lần chạy (Jaccard ≈ 0,85); ba nhóm còn lại thay đổi theo mẫu. GMM xác nhận: Hardcore ≈ 1% giáp ranh, ba nhóm kia 21–30%.
- Chi tiêu: tỉ lệ mua ≈ 20% ở mọi nhóm (χ², p = 0,36); Gradient Boosting dự báo mua hàng AUC 0,50. Không có nhóm High-Spender.
- Cây quyết định 3 tầng gán phân khúc chính xác ≈ 91% với 8 quy tắc.

## Đầu ra
- `notebooks/gamer_clustering.ipynb` (sinh từ `tools/build_notebook.py`), chạy được trên máy và Colab.
- `reports/REPORT.md`, `reports/figures/` (19 biểu đồ, 5 bảng CSV).

## Ngoài phạm vi
Dashboard, triển khai mô hình thành dịch vụ, tinh chỉnh siêu tham số cho mô hình có giám sát, DBSCAN/HDBSCAN (thử nhanh: HDBSCAN không tìm thấy cụm nào, toàn bộ là nhiễu, nhất quán với Hopkins thấp).
