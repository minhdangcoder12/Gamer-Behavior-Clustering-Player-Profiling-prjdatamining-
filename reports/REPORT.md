# Báo cáo: Gamer Behavior Clustering & Player Profiling

## Tóm tắt

Đề tài phân khúc 40.034 người chơi game online theo lối chơi bằng **Agglomerative Hierarchical Clustering kết hợp K-Means**. Ba đặc trưng (giờ chơi/tuần, level, thành tựu) chia người chơi thành **4 phân khúc**: Hardcore, Veteran Grinders, Achievement Hunters, Casual. Phương pháp kết hợp đạt **Silhouette 0,276** và **Davies-Bouldin 1,059**, tốt hơn rõ Agglomerative đơn thuần và ngang K-Means++.

Các phân tích bổ sung cho thấy dữ liệu chỉ có xu hướng phân cụm yếu (Hopkins 0,58). Nhóm **Hardcore là phân khúc vững nhất**, được tìm lại ở mọi lần chạy, còn ba nhóm kia có ranh giới mềm. Tỉ lệ mua hàng như nhau ở mọi nhóm, nên dataset **không có nhóm High-Spender**. Chiến lược monetization được đề xuất cho từng nhóm, kèm mức độ tin cậy và bộ quy tắc gán phân khúc cho người chơi mới.

## 1. Đề bài và dữ liệu

**Đề bài:** trích xuất chỉ số lối chơi (playtime, win/loss ratio, item collection rate, skill score), phân khúc người chơi bằng Agglomerative Hierarchical Clustering kết hợp K-Means, đánh giá bằng Silhouette Score và Davies-Bouldin Index, đề xuất chiến lược monetization.

**Dữ liệu:** [Predict Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset) (Rabie El Kharoua, Kaggle, giấy phép CC BY 4.0). 40.034 người chơi, 13 cột, không có giá trị thiếu, không trùng lặp.

## 2. Khám phá dữ liệu

- Biến phân loại chia gần như đều tuyệt đối (5 thể loại game mỗi loại ≈ 20%, độ khó 50/30/20) và các biến số phân bố gần đều (hình `01_distributions.png`): dấu hiệu **dữ liệu tổng hợp**.
- Tương quan với `EngagementLevel` (hình `02_engagement_corr.png`):

| Biến | Tương quan |
|---|---|
| SessionsPerWeek | 0,606 |
| AvgSessionDurationMinutes | 0,477 |
| AchievementsUnlocked | 0,061 |
| PlayerLevel | 0,059 |
| InGamePurchases | 0,008 |
| **PlayTimeHours** | **−0,002** |

→ `PlayTimeHours` không phản ánh mức độ chơi. Thay bằng **WeeklyPlayHours = SessionsPerWeek × AvgSessionDurationMinutes / 60**, tương quan **0,773**.

- `EngagementLevel` gần như là **quy tắc ngưỡng** theo giờ chơi/tuần (hình `04_engagement_rule.png`): dưới 5 giờ ≈ 83% Low; 5–20 giờ ≈ 90% Medium; trên 25 giờ ≈ 93% High. Nhãn này không mang thêm thông tin, nên chỉ dùng để kiểm chứng.

## 3. Đặc trưng

| Đề bài | Đặc trưng | Ghi chú |
|---|---|---|
| Playtime | WeeklyPlayHours | đặc trưng tạo mới |
| Skill score | PlayerLevel | proxy |
| Item collection rate | AchievementsUnlocked | proxy |
| Win/loss ratio | — | không có trong dataset |

Ba đặc trưng độc lập với nhau (|r| ≈ 0, hình `03_feature_corr.png`). `InGamePurchases` và biến nhân khẩu học không đưa vào phân cụm, chỉ dùng để mô tả hồ sơ.

**Ablation** (hình `06_feature_ablation.png`, Hybrid k = 4):

| Bộ đặc trưng | Silhouette | Davies-Bouldin |
|---|---|---|
| 5 biến gốc | 0,144 | 1,882 |
| PlayTimeHours thay giờ/tuần | 0,268 | 1,178 |
| Đã chọn + InGamePurchases | 0,250 | 1,253 |
| **Đã chọn (giờ/tuần, level, thành tựu)** | **0,276** | **1,059** |

## 4. Dữ liệu có cấu trúc cụm không?

**Hopkins statistic** (hình `05_hopkins.png`):

| Dữ liệu | Hopkins |
|---|---|
| Ngẫu nhiên đều (không có cụm) | 0,50 |
| **Dữ liệu người chơi** | **0,58** |
| 4 cụm tách biệt (cụm rõ) | 0,97 |

Dữ liệu chỉ có xu hướng phân cụm yếu (dưới ngưỡng 0,75), nên Silhouette vừa phải là giới hạn của dữ liệu chứ không phải của phương pháp.

## 5. Phương pháp

1. Chuẩn hóa 3 đặc trưng bằng `StandardScaler`.
2. **Agglomerative (Ward)** trên mẫu ngẫu nhiên 5.000 người (ma trận khoảng cách 40k điểm cần ≈ 6 GB RAM) để chọn k và lấy tâm cụm.
3. **Hybrid:** tâm cụm Agglomerative làm điểm khởi tạo cho **K-Means trên toàn bộ 40.034 người**.
4. So sánh với Agglomerative đơn thuần và K-Means++ (`n_init=10`).
5. Silhouette tính trên mẫu 10.000 người; Davies-Bouldin trên toàn bộ. `SEED = 42`.

**Chọn linkage** (hình `07_linkage_comparison.png`, k = 4, mẫu 5k):

| Linkage | Silhouette | Davies-Bouldin | Cụm nhỏ nhất |
|---|---|---|---|
| **Ward** | 0,212 | 1,204 | 968 |
| Complete | 0,229 | 1,182 | 927 |
| Average | 0,200 | 1,239 | 79 |
| Single | 0,273 | 0,494 | 1 |

Single có điểm cao nhưng dồn gần hết người chơi vào một cụm. Ward và Complete đều cân đối và cho kết quả Hybrid gần bằng nhau (Silhouette 0,276 so với 0,278). Chọn **Ward** vì cùng hàm mục tiêu (tổng phương sai trong cụm) với K-Means.

**Chọn k** (hình `08_choose_k.png`, `09_dendrogram.png`):

| k | 2 | 3 | **4** | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| Silhouette | 0,235 | 0,230 | **0,212** | 0,206 | 0,223 | 0,228 | 0,211 |
| Davies-Bouldin | 1,635 | 1,395 | **1,204** | 1,106 | 1,187 | 1,097 | 1,193 |

Silhouette dao động hẹp (0,21–0,24) nên không phân định được k. Davies-Bouldin giảm mạnh tới **k = 4** rồi chững lại (điểm khuỷu). k = 2 chỉ tách "chơi nhiều / chơi ít", quá thô cho kinh doanh.

## 6. Đánh giá

(hình `10_method_comparison.png`)

| Phương pháp (k = 4) | Silhouette ↑ | Davies-Bouldin ↓ | Số lần chạy |
|---|---|---|---|
| Agglomerative (mẫu 5k) | 0,212 | 1,204 | 1 |
| K-Means++ (`n_init=10`) | 0,281 | 1,059 | 10 |
| **Hybrid Agglomerative → K-Means** | **0,276** | **1,059** | **1** |

- Hybrid **tốt hơn rõ** Agglomerative đơn thuần (Silhouette +30%, DB −12%) nhờ K-Means tinh chỉnh tâm cụm trên toàn bộ dữ liệu.
- Hybrid **ngang** K-Means++ với một lần chạy thay vì 10, và tâm khởi tạo giải thích được qua dendrogram.
- ARI giữa Hybrid và K-Means++ chỉ 0,56: hai cách chia khác nhau dù điểm gần bằng nhau, vì dữ liệu không có ranh giới tự nhiên rõ.
- Silhouette ≈ 0,28 là mức cấu trúc yếu–trung bình; chỉ ≈ 5% người chơi có silhouette âm.

### Độ ổn định (hình `11_stability.png`)

Chạy lại Hybrid với 10 mẫu 5.000 người khác nhau, và K-Means khởi tạo ngẫu nhiên với 10 seed:

| | ARI so với kết quả chính (TB) | Silhouette (TB) | Trùng khớp nhóm Hardcore (Jaccard TB) |
|---|---|---|---|
| Hybrid (mẫu khác) | 0,50 | 0,279 | 0,85 |
| K-Means khởi tạo ngẫu nhiên | 0,63 | 0,278 | 0,89 |

- **Hardcore** được tìm lại ở mọi lần chạy (tâm cụm tụ một chỗ): phân khúc vững nhất.
- Ba nhóm còn lại thay đổi theo mẫu vì vùng level × thành tựu phân bố đều, không có ranh giới tự nhiên.
- Hybrid không ổn định hơn K-Means ngẫu nhiên khi đổi mẫu: sự bất ổn nằm ở dữ liệu. Giá trị của Hybrid là tiết kiệm (1 lần chạy) và giải thích được.

## 7. Hồ sơ phân khúc

(hình `12_segment_profiles.png`, `13_segment_fingerprint.png`, `14_pca_clusters.png`, `15_segment_demographics.png`)

| Phân khúc | Tỉ trọng | Giờ/tuần | Level | Thành tựu | Engagement High | Tỉ lệ mua | Silhouette |
|---|---|---|---|---|---|---|---|
| **Hardcore** | 19% | **36,1** | 50 | 23 | **92%** | 20% | 0,19 |
| **Veteran Grinders** | 26% | 10,2 | **77,7** | 33 | 13% | 21% | 0,31 |
| **Achievement Hunters** | 25% | 10,2 | 23,5 | **34,8** | 11% | 20% | 0,34 |
| **Casual** | 30% | 9,6 | 47 | 9 | 8% | 20% | 0,25 |

- **Hardcore:** chơi gấp ~3,5 lần các nhóm khác; 92% engagement High, nên nhãn có sẵn xác nhận độc lập đây là nhóm gắn bó nhất. Silhouette thấp nhất vì nhóm trải rộng theo level và thành tựu, dù tách rõ theo giờ chơi.
- **Veteran Grinders:** level cao nhất, chơi vừa phải: tích lũy tiến trình lâu dài.
- **Achievement Hunters:** level thấp nhưng nhiều thành tựu nhất: thích hoàn thành mục tiêu, sưu tập.
- **Casual:** nhóm lớn nhất, ít thành tựu, engagement thấp.
- Tuổi, khu vực, độ khó, thể loại game phân bố như nhau ở mọi nhóm: phân khúc thuần theo hành vi.

## 8. Chi tiêu: có nhóm High-Spender không?

(hình `16_purchase_rate.png`)

Tỉ lệ mua ≈ 20% ở mọi phân khúc (19,5%–20,5%). Kiểm định χ² = 3,21, p = 0,36: không có khác biệt có ý nghĩa. Dataset chỉ ghi có/không mua, không có số tiền, nên **không xác định được nhóm High-Spender**. Hành vi chơi không dự báo được việc chi tiêu.

## 9. Mô hình bổ sung

| Mô hình | Câu hỏi | Kết quả |
|---|---|---|
| **Gaussian Mixture Model** (hình `17_gmm_uncertainty.png`) | Ranh giới giữa các nhóm mềm đến mức nào? | ≈ 20% người chơi giáp ranh (xác suất thuộc nhóm < 0,7). Hardcore chỉ ≈ 1%, ba nhóm kia 21–30%: xác nhận độc lập kết quả độ ổn định. Silhouette 0,21, thấp hơn Hybrid. |
| **Cây quyết định** (hình `18_segment_tree.png`) | Gán phân khúc cho người chơi mới thế nào? | Cây 3 tầng (8 quy tắc) chính xác ≈ 91%; cây 5 tầng ≈ 95% (31 quy tắc). |
| **Gradient Boosting** (hình `19_supervised_checks.png`) | Các phát hiện có đứng vững với mô hình mạnh? | Engagement: độ chính xác 92,7% (đoán lớp phổ biến nhất: 48,4%), giờ chơi/tuần chiếm gần hết importance. Mua hàng: ROC-AUC 0,502, bằng đoán ngẫu nhiên. |

**Quy tắc gán phân khúc (cây 3 tầng):**
1. Giờ chơi/tuần > ~26 → **Hardcore**.
2. Ngược lại, thành tựu ≤ ~18 → **Casual**.
3. Ngược lại, level ≤ ~51 → **Achievement Hunters**; level > ~51 → **Veteran Grinders**.

## 10. Chiến lược monetization

| Phân khúc | Mục tiêu | Chiến lược | Độ tin cậy |
|---|---|---|---|
| **Hardcore** (19%) | Tối đa giá trị vòng đời, giữ chân | Season/battle pass, subscription, cosmetic cao cấp, nội dung end-game trả phí. Không giảm giá sâu vì nhóm đã gắn bó. | **Cao** |
| **Veteran Grinders** (26%) | Ngăn rời bỏ khi hết nội dung | DLC, chế độ chơi mới, vật phẩm uy tín (prestige skins), sự kiện giới hạn thời gian. | Trung bình |
| **Achievement Hunters** (25%) | Biến động lực sưu tập thành doanh thu | Bộ sưu tập (collection sets), thưởng hoàn thành bộ, loot box có pity minh bạch, XP booster. | Trung bình |
| **Casual** (30%) | Chuyển đổi lần mua đầu | Starter pack giá rẻ, rewarded ads, ưu đãi lần đầu, thưởng đăng nhập hằng ngày. | Trung bình |

**Áp dụng chung:**
- Gắn phân khúc cho người chơi mới bằng quy tắc cây quyết định (≈ 91% chính xác), không cần chạy lại phân cụm.
- Ba nhóm không phải Hardcore có ranh giới mềm, nên cá nhân hóa theo hai trục liên tục level và thành tựu; người chơi giáp ranh nhận ưu đãi kết hợp của hai nhóm lân cận.
- **Đòn bẩy lớn nhất:** engagement do giờ chơi/tuần quyết định, nên các tính năng tăng số phiên hoặc độ dài phiên (sự kiện hằng ngày, nhiệm vụ tuần) sẽ đẩy người chơi lên nhóm Hardcore.
- ~20% người đã từng mua trong mỗi nhóm là tập khách hàng ưu tiên; đo hiệu quả bằng A/B test trên tỉ lệ chuyển đổi và ARPU.

## 11. Kết luận

1. Bộ 3 đặc trưng (giờ/tuần, level, thành tựu) tốt nhất theo ablation. `PlayTimeHours` là nhiễu; `EngagementLevel` gần như suy ra từ giờ chơi/tuần.
2. Dữ liệu có xu hướng phân cụm yếu (Hopkins 0,58).
3. Hybrid Agglomerative (Ward) → K-Means đạt Silhouette 0,276, Davies-Bouldin 1,059: tốt hơn rõ Agglomerative đơn thuần, ngang K-Means++ với một lần chạy.
4. Bốn phân khúc khác nhau thuần túy ở hành vi; Hardcore là phân khúc vững nhất (độ ổn định và GMM cùng xác nhận).
5. Không có nhóm High-Spender (χ², p = 0,36; Gradient Boosting AUC 0,50).
6. Cây quyết định 3 tầng cho bộ quy tắc gán phân khúc dùng được trong vận hành.

## 12. Hạn chế và hướng phát triển

**Hạn chế:**
- Không có win/loss ratio; skill score và item collection rate dùng proxy.
- `InGamePurchases` nhị phân, không đo được mức chi tiêu.
- Dữ liệu tổng hợp với xu hướng cụm yếu; ba phân khúc không phải Hardcore không ổn định giữa các lần lấy mẫu. Kết luận cần kiểm chứng trên dữ liệu thật.

**Hướng phát triển:**
- Bổ sung dữ liệu giao dịch (số tiền, tần suất mua) để xác định High-Spenders và ước tính doanh thu từng nhóm.
- Theo dõi người chơi qua nhiều thời điểm để đo chuyển dịch giữa các phân khúc.
- Kiểm chứng quy tắc cây quyết định và các chiến lược bằng A/B test trên người chơi thật.

## 13. Tái lập kết quả

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/gamer_clustering.ipynb
```

Hoặc mở notebook trên Google Colab → Run All (dataset tự tải qua `kagglehub`). Notebook chạy khoảng 2–3 phút trên CPU; 19 biểu đồ và 5 bảng CSV xuất ra `reports/figures/`.
