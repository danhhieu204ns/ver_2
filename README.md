python -m pip install -r requirements.txt
bash scripts/run_all_baselines.sh
bash scripts/run_dataset_ablation.sh
bash scripts/run_loss_ablation.sh
python scripts/check_experiment_protocol.py

# Kế hoạch cuối cùng: HN-SARD cho Nine-Dash Line Detection

## 0. Định vị lại bài báo

Bài cũ có câu chuyện chính là:

> Áp dụng Faster R-CNN kết hợp DINOv2-based KD để phát hiện nine-dash line.

Nhưng bản cũ còn yếu ở dataset nhỏ, split đơn giản 90/10, chưa có negative in-domain/out-domain rõ ràng, và chỉ báo mAP/AR .

Bài mới nên định vị lại thành:

> **A real-image benchmark and hard-negative scale-aware region distillation framework for tiny, thin, cartographic-symbol moderation.**

Tức là bài không chỉ nói “tôi dùng model A”, mà nói:

1. Bài toán này khó vì object **nhỏ, mảnh, đứt đoạn, dễ nhầm với các đường nét bản đồ**.
2. Dataset mới có **positive, negative in-domain, negative out-domain**, toàn bộ là ảnh thật.
3. Method mới tập trung xử lý đúng hai vấn đề cốt lõi:

   * miss tiny/small object;
   * false positive trên bản đồ sạch.

---

# 1. Contribution cuối cùng của bài báo

Nên chốt 3 contribution chính:

## Contribution 1 — Real-image benchmark

Xây dựng benchmark phát hiện nine-dash line gồm **100% ảnh thật**, có phân chia:

* positive images;
* negative in-domain: bản đồ/ảnh địa lý dễ gây nhầm;
* negative out-domain: ảnh ngoài miền;
* train/val/test stratified theo kích thước object.

Dataset hiện tại:

| Split | Positive images | Negative images | Total images | Positive objects |
| ----- | --------------: | --------------: | -----------: | ---------------: |
| Train |           1,642 |           3,499 |        5,141 |            1,837 |
| Val   |             352 |             750 |        1,102 |              402 |
| Test  |             351 |             750 |        1,101 |              397 |

Trong test set:

| Size group | Images | Objects |
| ---------- | -----: | ------: |
| Large      |    201 |     202 |
| Medium     |     93 |     104 |
| Small      |     43 |      67 |
| Tiny       |     14 |      24 |

Cách viết cần thận trọng:

> Tiny objects are rare in real-world nine-dash line imagery; therefore, we report AP for tiny objects with confidence intervals and complement quantitative results with detailed qualitative error analysis.

Không claim “SOTA tiny object detection”.

---

## Contribution 2 — HN-SARD method

Tên method chốt:

> **HN-SARD: Hard-Negative Scale-Aware Region Distillation**

Ý tưởng chính:

* dùng Faster R-CNN R50-FPN làm student detector;
* dùng DINOv2 như **self-supervised visual foundation teacher**;
* dùng region crop + context resize để DINOv2 nhìn rõ tiny/thin object;
* thêm hai loss chính:

  * scale-aware positive region distillation;
  * online hard-negative contrastive distillation.

---

## Contribution 3 — Moderation-oriented evaluation

Không chỉ báo mAP. Bài phải báo cả metric theo nhu cầu kiểm duyệt:

* object-level AP;
* AP theo kích thước object;
* image-level AUROC/AP;
* FPR@95TPR;
* in-domain FPR;
* FROC curve;
* error taxonomy.

Đây là điểm giúp bài khác object detection thông thường.

---

# 2. Method cuối cùng

## 2.1. Detector chính

Chốt dùng:

> **Faster R-CNN R50-FPN**

Lý do:

* phù hợp small object hơn nhiều one-stage detector trong nhiều trường hợp;
* dễ can thiệp vào RoI features;
* dễ lấy proposals để làm hard-negative mining;
* phù hợp với hướng region distillation.

Các baseline hiện đại vẫn cần chạy để so sánh:

* Faster R-CNN R50-FPN;
* Faster R-CNN R101-FPN;
* YOLOv10 hoặc YOLOv11;
* RT-DETR;
* Faster R-CNN + old KD;
* proposed HN-SARD.

---

## 2.2. Teacher DINOv2

Cách dùng DINOv2 phải sửa so với bản cũ.

Không viết:

> DINOv2 vision-language model.

Viết:

> DINOv2 is used as a frozen self-supervised visual foundation teacher.

Với mỗi GT/proposal region:

1. Crop vùng bbox.
2. Mở rộng context theo hệ số (c), ưu tiên (c = 1.5) hoặc (c = 2.0).
3. Resize crop lên 224×224.
4. Đưa qua DINOv2.
5. Lấy CLS token hoặc average patch token làm teacher embedding.

Lý do:

* nếu dùng ảnh full-resolution, DINOv2 patch size có thể quá thô với tiny object;
* crop + resize giúp tiny object trở nên đủ lớn;
* context giúp phân biệt nine-dash line với đường đứt đoạn thông thường.

---

# 3. Loss function chốt

Tổng loss:

[
L = L_{det} + \lambda_{pos}L_{pos} + \lambda_{con}L_{con}
]

Trong giai đoạn đầu, **không nên đưa RPN attention distillation vào main method**. Nếu có thời gian, để supplementary hoặc future work.

---

## 3.1. Detection loss

[
L_{det} = L_{rpn-cls} + L_{rpn-reg} + L_{roi-cls} + L_{roi-reg}
]

Đây là loss chuẩn của Faster R-CNN.

---

## 3.2. Scale-aware positive region distillation

Với positive RoI (r_i), student embedding là:

[
z_i^S = f_S(r_i)
]

Teacher embedding từ context crop là:

[
z_i^T = f_T(crop(g_i))
]

Sau đó L2-normalize:

[
\hat{z}_i^S = \frac{z_i^S}{|z_i^S|_2}, \quad
\hat{z}_i^T = \frac{z_i^T}{|z_i^T|_2}
]

Dùng cosine distance:

[
d_i = 1 - cos(\hat{z}_i^S, \hat{z}_i^T)
]

Scale-aware weight dùng bucket:

| Object size | Weight |
| ----------- | -----: |
| Large       |    1.0 |
| Medium      |    1.2 |
| Small       |    1.5 |
| Tiny        |    2.0 |

Loss:

[
L_{pos} = \frac{1}{\sum_{i \in P} \alpha_i}
\sum_{i \in P}
\alpha_i \cdot
\left(1 - cos(\hat{z}_i^S, \hat{z}_i^T)\right)
]

Điểm quan trọng là **normalize bằng tổng weight**. Như vậy loss không bị bùng khi batch có nhiều tiny/small objects.

---

## 3.3. Online hard-negative contrastive distillation

Đây là linh hồn của novelty.

### Chọn hard negative

Với ảnh positive:

[
HN = {r_j \mid IoU(r_j, G) < \theta_n}
]

Với ảnh negative:

[
HN = \text{top-K proposals by objectness/class score}
]

Chỉ lấy top-K proposal nguy hiểm nhất.

Thông số khởi đầu:

| Tham số                    | Giá trị đề xuất |
| -------------------------- | --------------: |
| (\theta_n)                 |    0.1 hoặc 0.3 |
| top-K hard negatives/image |              16 |
| score threshold            |   0.05 hoặc 0.1 |
| temperature (\tau)         |   0.07 hoặc 0.1 |

### Warm-up

Không bật (L_{con}) ngay từ đầu.

Chốt:

* epoch 1–5: train detector + (L_{pos});
* từ epoch 6 trở đi: bật (L_{con}).

Lý do: đầu training, proposal còn rác. Nếu lấy hard negative quá sớm, contrastive loss sẽ học từ nền ngẫu nhiên.

### Contrastive loss

Với positive RoI (i):

[
L_{con} =
-\frac{1}{|P|}
\sum_{i \in P}
\log
\frac{
\exp(cos(z_i^S, z_i^T)/\tau)
}{
\exp(cos(z_i^S, z_i^T)/\tau)
+
\sum_{j \in HN}
\exp(cos(z_i^S, h_j^T)/\tau)
}
]

Trong đó (h_j^T) là teacher embedding của hard-negative crop.

Ý nghĩa:

* kéo positive proposal về gần representation của nine-dash line;
* đẩy positive proposal ra xa các vùng dễ nhầm;
* đặc biệt giảm false positive trên negative in-domain.

---

# 4. Dataset analysis cần làm

## 4.1. Statistics bắt buộc

Trong paper cần có:

* số ảnh theo split;
* số object theo split;
* số negative in-domain/out-domain;
* phân phối bbox area;
* phân phối bbox width/height;
* phân phối object scale: tiny/small/medium/large.

## 4.2. UMAP/t-SNE visualization

Nên làm một hình UMAP hoặc t-SNE dùng DINOv2 feature.

Mục tiêu không phải để chứng minh model, mà để chứng minh dataset đa dạng.

Nên vẽ:

* positive large/medium/small/tiny;
* negative in-domain;
* negative out-domain.

Nếu hình cho thấy negative in-domain nằm gần positive hơn out-domain, đó là bằng chứng rất mạnh:

> In-domain negatives are visually closer to positive samples than out-domain negatives, making them more challenging and more relevant for moderation.

---

# 5. Experiment plan cuối cùng

## Experiment 1 — Main comparison

Bảng chính:

| Method                | AP | AP50 | AP75 | AP_tiny | AP_small | AP_medium | AP_large | FPR@95TPR | In-domain FPR |
| --------------------- | -: | ---: | ---: | ------: | -------: | --------: | -------: | --------: | ------------: |
| Faster R-CNN R50-FPN  |    |      |      |         |          |           |          |           |               |
| Faster R-CNN R101-FPN |    |      |      |         |          |           |          |           |               |
| YOLOv10/YOLOv11       |    |      |      |         |          |           |          |           |               |
| RT-DETR               |    |      |      |         |          |           |          |           |               |
| Old KD                |    |      |      |         |          |           |          |           |               |
| HN-SARD               |    |      |      |         |          |           |          |           |               |

Chỉ cần HN-SARD thắng ở mọi metric thì quá tốt. Nhưng nếu không thắng AP tổng, vẫn có thể thắng ở:

* FPR@95TPR;
* in-domain FPR;
* AP_small/AP_tiny;
* FROC.

---

## Experiment 2 — Dataset ablation

Bảng này rất quan trọng.

| Training data                  | AP | AP50 | FPR in-domain | FPR out-domain | FPR@95TPR |
| ------------------------------ | -: | ---: | ------------: | -------------: | --------: |
| Positive only                  |    |      |               |                |           |
| Positive + negative out-domain |    |      |               |                |           |
| Positive + negative in-domain  |    |      |               |                |           |
| Positive + both negatives      |    |      |               |                |           |

Mục tiêu:

* chứng minh positive-only dễ false positive;
* negative out-domain không đủ;
* negative in-domain là yếu tố quan trọng nhất để giảm báo nhầm trên bản đồ sạch.

---

## Experiment 3 — Method ablation

| Variant                | AP | AP_tiny | AP_small | FPR in-domain | FPR@95TPR |
| ---------------------- | -: | ------: | -------: | ------------: | --------: |
| Baseline Faster R-CNN  |    |         |          |               |           |
| + (L_{pos})            |    |         |          |               |           |
| + scale-aware weight   |    |         |          |               |           |
| + hard-negative mining |    |         |          |               |           |
| + warm-up strategy     |    |         |          |               |           |
| Full HN-SARD           |    |         |          |               |           |

Mục tiêu:

* (L_{pos}) giúp tăng localization/recall;
* scale-aware weight giúp small/tiny tốt hơn;
* (L_{con}) giúp giảm FPR in-domain;
* warm-up giúp training ổn định hơn.

---

## Experiment 4 — Context crop ablation

| Teacher crop strategy | AP | AP_tiny | AP_small | FPR in-domain |
| --------------------- | -: | ------: | -------: | ------------: |
| bbox crop only        |    |         |          |               |
| 1.5× context crop     |    |         |          |               |
| 2.0× context crop     |    |         |          |               |
| 3.0× context crop     |    |         |          |               |

Tôi dự đoán 1.5× hoặc 2.0× sẽ tốt nhất. 3.0× có thể đưa quá nhiều nền, làm teacher embedding loãng.

---

## Experiment 5 — Robustness test

Tạo robustness set từ test image thật, không cần synthetic train.

Các biến đổi:

* JPEG compression;
* Gaussian blur;
* low-resolution resize;
* brightness/contrast shift;
* partial occlusion nhẹ.

Bảng:

| Method          | Clean AP | JPEG AP | Blur AP | Low-res AP | Brightness AP |
| --------------- | -------: | ------: | ------: | ---------: | ------------: |
| Faster R-CNN    |          |         |         |            |               |
| YOLOv10/YOLOv11 |          |         |         |            |               |
| HN-SARD         |          |         |         |            |               |

Không cần quá nhiều biến đổi. Chọn 3–4 loại chính là đủ.

---

# 6. Metrics chốt

## Object-level metrics

Bắt buộc:

* AP@[.50:.95]
* AP50
* AP75
* AP_tiny
* AP_small
* AP_medium
* AP_large
* AR@100

## Image-level moderation metrics

Với mỗi ảnh:

[
score(I) = \max_{b \in B_I} conf(b)
]

Sau đó báo:

* image-level AUROC;
* image-level AP;
* FPR@95TPR;
* Recall@FPR=1%;
* Recall@FPR=5%;
* in-domain FPR;
* out-domain FPR.

## FROC curve

Nên có FROC:

* trục X: average false positives per image;
* trục Y: recall.

FROC rất hợp với bài toán cần cân bằng giữa bỏ sót và báo nhầm.

---

# 7. Bootstrap confidence interval

Vì test chỉ có 397 objects và tiny chỉ có 24 objects, nên cần bootstrap.

Chốt làm:

* bootstrap theo image-level, không theo object-level;
* sample lại test images với replacement;
* lặp 1,000 lần;
* báo mean và 95% confidence interval.

Báo cho:

* AP;
* AP_tiny;
* AP_small;
* FPR@95TPR;
* in-domain FPR.

Cách viết:

> Since tiny nine-dash line instances are rare in real-world data, we report bootstrap confidence intervals to assess the stability of the observed improvements.

---

# 8. Error analysis cuối cùng

Phải làm kỹ phần này.

## Taxonomy lỗi

| Error type                | Mô tả                                             |
| ------------------------- | ------------------------------------------------- |
| Tiny miss                 | bỏ sót object quá nhỏ/mờ                          |
| Low-contrast miss         | đường lưỡi bò gần màu nền                         |
| Fragmented detection      | detect từng đoạn rời rạc                          |
| Mislocalization           | bbox lệch/quá rộng/quá hẹp                        |
| In-domain false positive  | nhầm coastline, river, dashed boundary, grid line |
| Out-domain false positive | nhầm đường cong/texture ngoài miền                |

## Figure nên có

Một figure 4 cột:

1. Image;
2. Ground truth;
3. Baseline prediction;
4. HN-SARD prediction.

Chia thành các hàng:

* tiny object;
* low contrast;
* in-domain false positive;
* fragmented line;
* hard negative map.

## Pie chart

Làm 2 pie chart:

* error distribution của baseline;
* error distribution của HN-SARD.

Nếu HN-SARD giảm rõ nhóm **in-domain false positive**, đây là bằng chứng rất mạnh.

---

# 9. Câu hỏi phản biện: nếu mAP thấp hơn YOLOv10 nhưng FPR in-domain thấp hơn 50% thì biện luận thế nào?

Câu trả lời nên là:

> Trong bài toán kiểm duyệt, mục tiêu không chỉ là tối đa hóa AP object-level, mà là vận hành ở vùng high-recall, low-false-alarm. Một model có mAP cao hơn nhưng gây báo nhầm nhiều trên bản đồ sạch sẽ tạo chi phí kiểm duyệt lớn, gây phiền hà cho người dùng, và có thể làm giảm độ tin cậy của hệ thống. Vì vậy, chúng tôi đánh giá thêm image-level moderation metrics như FPR@95TPR, Recall@FPR=1%, và in-domain FPR. Kết quả cho thấy HN-SARD phù hợp hơn với deployment scenario vì giảm đáng kể false positives trên in-domain negatives trong khi vẫn duy trì recall cạnh tranh.

Nói ngắn gọn:

> YOLO tốt hơn về object-level AP, nhưng HN-SARD tốt hơn về operational moderation risk.

Trong paper, nên viết:

* AP là metric kỹ thuật;
* FPR@95TPR là metric vận hành;
* in-domain FPR là metric đặc thù của bài toán kiểm duyệt bản đồ.

Nếu HN-SARD có mAP thấp hơn một chút nhưng giảm 50% in-domain FPR, vẫn có thể claim:

> HN-SARD provides a better precision-safety trade-off for high-recall moderation.

Không nên claim “overall better detector”. Hãy claim:

> more suitable for moderation-oriented deployment.

---

# 10. Cấu trúc paper cuối cùng

## Abstract

Nên nhấn mạnh:

* nine-dash line detection là bài toán tiny/thin cartographic symbol moderation;
* dataset mới 100% real images;
* negative in-domain là thách thức chính;
* đề xuất HN-SARD;
* kết quả chính: AP, FPR@95TPR, in-domain FPR.

## 1. Introduction

Luồng viết:

1. Nine-dash line xuất hiện trong digital content và cần kiểm duyệt tự động.
2. Khó khăn không chỉ là detect object, mà là detect **thin, fragmented, tiny cartographic symbols**.
3. Negative in-domain rất nguy hiểm vì nhiều bản đồ sạch có coastline, dashed lines, grid lines giống nine-dash line.
4. Bài cũ/detector thông thường chưa đánh giá đủ false positive trên in-domain negatives.
5. Đề xuất real-image benchmark + HN-SARD.

## 2. Related Work

Chia thành:

1. Object detection for small/thin objects;
2. Knowledge distillation for object detection;
3. Hard negative mining / contrastive learning;
4. Visual foundation models for dense recognition;
5. Content moderation / harmful visual content detection.

## 3. Dataset

Nội dung:

* data source;
* annotation protocol;
* real-image-only design;
* positive/negative definition;
* in-domain vs out-domain negative;
* split strategy;
* bbox size distribution;
* UMAP/t-SNE visualization;
* limitations.

## 4. Method

Nội dung:

1. Overview;
2. Context-aware DINOv2 region teacher;
3. Scale-aware positive region distillation;
4. Online hard-negative mining;
5. Hard-negative contrastive distillation;
6. Training objective.

## 5. Experiments

Nội dung:

1. Implementation details;
2. Main comparison;
3. Dataset ablation;
4. Method ablation;
5. Context crop ablation;
6. Robustness test;
7. Bootstrap confidence interval.

## 6. Error Analysis and Discussion

Nội dung:

* qualitative results;
* error taxonomy;
* FPR discussion;
* case where YOLO has higher AP but worse in-domain FPR;
* deployment implication.

## 7. Limitations

Nên thành thật:

* tiny test set còn ít;
* dataset nhạy cảm nên có thể khó public toàn bộ;
* DINOv2 teacher tăng chi phí training;
* method hiện tập trung vào image-level, chưa mở rộng sang video/document PDF pipeline;
* object definition có thể khó trong trường hợp nine-dash line bị vỡ mảnh.

## 8. Conclusion

Chốt lại:

* benchmark ảnh thật;
* method HN-SARD;
* giảm false positive in-domain;
* phù hợp moderation deployment.

---

# 11. Timeline thực thi

## Giai đoạn 1 — Baseline sạch, 1–2 tuần

Việc cần làm:

* chuẩn hóa dataset loader;
* train Faster R-CNN R50-FPN;
* train Faster R-CNN R101-FPN;
* chạy YOLOv10/YOLOv11 hoặc RT-DETR;
* viết script COCO metrics;
* viết script image-level FPR@95TPR.

Output:

* main baseline table version 1;
* confusion/FPR analysis;
* qualitative baseline errors.

---

## Giai đoạn 2 — Dataset ablation, 1 tuần

Chạy:

1. Positive only;
2. Positive + negative out-domain;
3. Positive + negative in-domain;
4. Positive + both negatives.

Output:

* bảng chứng minh vai trò của negative in-domain;
* figure false positive examples.

Script đã triển khai trong repo:

```bash
# Smoke test: chỉ kiểm tra tạo biến thể dataset và loader.
DRY_RUN=1 MAX_TRAIN_IMAGES=8 MAX_VAL_IMAGES=4 MAX_TEST_IMAGES=4 \
  bash scripts/run_dataset_ablation.sh

# Full run mặc định dùng Faster R-CNN R50 cho 4 biến thể.
bash scripts/run_dataset_ablation.sh
```

Các output chính:

* dataset biến thể: `results/dataset_ablation/data/<variant>`;
* training run: `results/dataset_ablation/runs/<variant>`;
* bảng Experiment 2: `results/dataset_ablation/tables/dataset_ablation_test.{csv,md}`;
* false-positive overlays: `results/dataset_ablation/errors/<variant>/overlays`.

Nếu chỉ muốn dựng lại bảng từ các run đã có:

```bash
.venv/bin/python scripts/summarize_dataset_ablation.py \
  --result-root results/dataset_ablation/runs \
  --output-dir results/dataset_ablation/tables \
  --split test
```

---

## Giai đoạn 3 — Implement HN-SARD core, 2–3 tuần

Script đã triển khai trong repo:

* `scripts/train_hnsard.py`: trainer HN-SARD dùng Faster R-CNN, DINOv2 context crop teacher, `L_pos`, scale-aware weight, hard-negative top-K mining, warm-up, và `L_con`;
* `scripts/run_loss_ablation.sh`: chạy loss ablation cho Experiment 3;
* `scripts/summarize_loss_ablation.py`: dựng bảng từ `final_metrics.json`.

Chạy smoke test trước để kiểm tra pipeline không cần tải DINOv2:

```bash
VARIANTS="baseline l_pos_scale_l_con" \
TEACHER_BACKEND=dummy \
EPOCHS_LOSS_ABLATION=1 \
MAX_TRAIN_IMAGES=8 MAX_VAL_IMAGES=4 MAX_TEST_IMAGES=4 \
BATCH_SIZE=1 EVAL_BATCH_SIZE=1 WORKERS=0 \
NO_PRETRAINED=1 MIN_SIZE=128 MAX_SIZE=256 \
SKIP_FINAL_EVAL=1 RUN_SUMMARY=0 COLLECT_ERRORS=0 \
ABLATION_ROOT=results/hnsard_smoke/loss_ablation \
  bash scripts/run_loss_ablation.sh
```

Chạy full loss ablation mặc định:

```bash
bash scripts/run_loss_ablation.sh
```

Lần chạy thật đầu tiên có thể tải `facebook/dinov2-small` qua Hugging Face. Nếu máy đã có cache và muốn ép offline, thêm `TEACHER_LOCAL_FILES_ONLY=1`.

Mặc định script chạy 5 biến thể:

1. `baseline`: detection loss only;
2. `l_pos`: thêm `L_pos`;
3. `l_pos_scale`: thêm `L_pos` + scale-aware weight;
4. `l_pos_scale_l_con`: thêm `L_pos` + scale-aware weight + `L_con` không warm-up;
5. `full_hnsard`: full HN-SARD với warm-up cho `L_con`.

Các output chính:

* training run: `results/hnsard_loss_ablation/runs/<variant>`;
* bảng Experiment 3: `results/hnsard_loss_ablation/tables/loss_ablation_test.{csv,md}`;
* false-positive overlays: `results/hnsard_loss_ablation/errors/<variant>/overlays`.

Nếu chỉ muốn chạy full HN-SARD một mình:

```bash
.venv/bin/python scripts/train_hnsard.py \
  --model fasterrcnn_r50 \
  --data-root data \
  --output-dir results/hnsard/full_hnsard \
  --epochs 50 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --workers 8 \
  --hflip-prob 0.5 \
  --aug-brightness 0.2 \
  --aug-saturation 0.2 \
  --aug-hue 0.015 \
  --lambda-pos 0.5 \
  --lambda-con 0.1 \
  --scale-aware \
  --contrastive-warmup-epochs 3 \
  --patience 15
```

Nếu chỉ muốn dựng lại bảng từ các run đã có:

```bash
.venv/bin/python scripts/summarize_loss_ablation.py \
  --result-root results/hnsard_loss_ablation/runs \
  --output-dir results/hnsard_loss_ablation/tables \
  --split test
```

---

## Giai đoạn 4 — Evaluation mở rộng, 1–2 tuần

Làm:

* bootstrap CI;
* FROC curve;
* robustness test;
* context crop ablation.

Output:

* đầy đủ bảng cho paper;
* hình biểu đồ.

---

## Giai đoạn 5 — Error analysis và viết paper, 2–3 tuần

Làm:

* đọc lỗi thủ công;
* phân loại lỗi;
* chọn ảnh minh họa;
* viết method;
* viết experiment;
* viết discussion;
* sửa lại introduction/related work.

Output:

* bản paper hoàn chỉnh;
* supplementary nếu cần.

---

# 12. Thứ tự ưu tiên nếu thiếu thời gian

Nếu không đủ thời gian, ưu tiên như sau:

## Bắt buộc phải có

1. Baseline Faster R-CNN R50-FPN.
2. YOLO hoặc RT-DETR baseline.
3. Dataset ablation positive-only vs positive + in-domain negative.
4. HN-SARD với (L_{pos}) + (L_{con}).
5. FPR@95TPR và in-domain FPR.
6. Error analysis.

## Có thì tốt

1. Bootstrap CI.
2. FROC curve.
3. UMAP/t-SNE dataset visualization.
4. Context crop ablation.
5. Robustness test.

## Có thể bỏ nếu không kịp

1. RPN attention distillation.
2. Multi-scale crop ensemble.
3. Relation distillation phức tạp.
4. Quá nhiều detector baseline.

---

# 13. Kết luận chốt

Kế hoạch cuối cùng nên đi theo hướng:

> **Dataset thật + negative in-domain + HN-SARD + moderation metrics.**

Không nên biến bài thành cuộc đua mAP đơn thuần. Bài của bạn mạnh nhất khi chứng minh được:

1. Nine-dash line detection không chỉ là object detection, mà là **sensitive visual moderation**.
2. Negative in-domain làm bài toán khó hơn nhiều vì gây false positive.
3. HN-SARD xử lý đúng vấn đề bằng hard-negative contrastive distillation.
4. Dù AP tổng không nhất thiết luôn cao nhất, method có thể tốt hơn ở **FPR@95TPR, in-domain FPR, và operational reliability**.

Thông điệp cuối cùng của paper nên là:

> **A detector for sensitive cartographic moderation should not only detect violations, but also avoid falsely flagging visually similar clean maps. HN-SARD is designed for this high-recall, low-false-positive setting.**

---

# 14. Early Stopping

Tất cả training script (Faster R-CNN, DETR, HN-SARD, YOLO) đều hỗ trợ early stopping dựa trên val mAP.

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--patience` | `15` | Dừng nếu val mAP không cải thiện sau N epoch eval liên tiếp. |
| `--patience 0` | — | Tắt early stopping, chạy đủ `--epochs`. |

## Cách hoạt động

1. Sau mỗi eval epoch, script so sánh val mAP với `best_map`.
2. Nếu cải thiện → lưu `best.pt`, reset counter.
3. Nếu không cải thiện → tăng `no_improve` lên 1.
4. Khi `no_improve >= patience` → in thông báo và dừng vòng lặp.
5. `best.pt` (checkpoint tốt nhất về val mAP) vẫn được load cho final eval.

## Tại sao `patience=15`?

Với `StepLR(step_size=5, gamma=0.1)`, LR giảm xuống `≤ 5e-6` sau epoch 15, tức là
các epoch từ 16 trở đi đóng góp rất ít. `patience=15` cho phép model hội tụ hoàn
toàn qua vài chu kỳ decay trước khi dừng.

## Override

```bash
# Tắt hoàn toàn early stopping
PATIENCE=0 bash scripts/run_all_baselines.sh

# Dùng patience khác cho loss ablation
PATIENCE_ABLATION=20 bash scripts/run_loss_ablation.sh

# Khi chạy trực tiếp
.venv/bin/python scripts/train_faster_rcnn_baseline.py \
  --model fasterrcnn_r50 \
  --data-root data \
  --output-dir results/baselines/fasterrcnn_r50 \
  --epochs 50 \
  --patience 15 \
  ...
```

## YOLO

YOLO nhận `patience=` trực tiếp từ framework của ultralytics:

```bash
.venv/bin/yolo detect train \
  model=yolo11s.pt \
  data=results/baselines/yolo_dataset/dataset.yaml \
  epochs=50 \
  patience=15 \
  ...
```

`run_yolo_baselines.sh` đã tự động truyền `patience="$PATIENCE"` vào lệnh trên.
