Dưới đây là kế hoạch cụ thể để nâng bài RIVF hiện tại thành một nghiên cứu mạnh hơn, khai thác tối đa dataset **11 ảnh** của bạn. Mục tiêu không nên chỉ là “mAP cao hơn”, mà là chứng minh hệ thống **robust với in-domain negatives, hard negatives, object nhỏ/mảnh, biến dạng phối cảnh và out-of-domain content**.

Bài gốc đang có nền tảng tốt: dataset cũ gồm **836 ảnh**, chia 752 train / 84 validation, dùng Faster R-CNN + DINOv2 KD với hai loss `L1` và `Lirm`, đạt **58.4% mAP / 66.2% AR**, cao hơn Faster R-CNN ResNet-50 baseline **52.2% mAP / 58.8% AR**.   Kế hoạch mới nên phát triển từ nền đó nhưng nâng cấp cả **dataset, method, evaluation protocol, ablation và error analysis**.

---

# I. Mục tiêu nghiên cứu mới

## 1. Narrative chính

Paper mới nên được định vị như sau:

> Chúng tôi xây dựng một benchmark lớn hơn và domain-aware hơn cho bài toán phát hiện nine-dash line, đồng thời đề xuất framework học robust dựa trên hard negative learning và DINOv2-based distillation để giảm false positive trên các mẫu bản đồ/sông/ngòi/đường xá giống nine-dash line.

Không nên chỉ viết:

> We improve Faster R-CNN with a larger dataset.

Mà nên viết:

> We address robust detection of small, thin, and visually ambiguous nine-dash-line patterns under in-domain and hard-negative distribution shifts.

## 2. Đóng góp kỳ vọng

Paper mới nên có 4 đóng góp:

1. **Large-scale domain-aware dataset**: 11k ảnh, gồm positive, in-domain negative, out-domain negative, synthetic positive và hard negative.
2. **Robust evaluation protocol**: đánh giá riêng theo positive, in-domain negative, out-domain negative, hard negative, small-object subset, distorted/perspective subset.
3. **Hard-negative-aware learning**: sampling hoặc loss để giảm nhầm sông/ngòi/đường xá/đường biên giống nine-dash line.
4. **Improved DINOv2 distillation**: mở rộng `L1 + IRM` cũ thành positive alignment + relational distillation + negative contrastive distillation.

---

# II. Giai đoạn 1 — Chuẩn hóa và thiết kế dataset

## Bước 1. Xây dựng taxonomy dataset

Bạn nên tổ chức dataset thành các nhóm rõ ràng:

```text
Dataset 11k
├── Positive real: 2.3k
│   └── ảnh thật có nine-dash line, có bbox annotation
│
├── Negative in-domain: 2.9k
│   └── bản đồ biển đảo / bản đồ Đông Nam Á / bản đồ biển Đông nhưng không có nine-dash line
│
├── Negative out-domain: 2.1k
│   └── ảnh không liên quan bản đồ, tài liệu, giao diện, ảnh thường
│
├── Synthetic positive: 2k3 (train-only)
│   └── positive được biến đổi phối cảnh, xoay, crop, blur, compression
│
└── Hard negative: 1k2 (train-only hoặc test riêng)
    └── sông ngòi, đường xá, đường biên, contour line, dashed line giống nine-dash line
```

Lưu ý quan trọng: nếu bạn muốn đánh giá hard-negative robustness, **không nên để toàn bộ hard negative là train-only**. Nên chia hard negative thành:

```text
Hard negative train: dùng để học
Hard negative test: giữ riêng, không xuất hiện trong train
```

Nếu toàn bộ hard negative chỉ dùng train, bạn sẽ không đo được khả năng generalize trên hard negative mới.

## Bước 2. Đặt metadata cho từng ảnh

Mỗi ảnh nên có file metadata dạng `.json` hoặc `.csv`:

```json
{
  "image_id": "xxx.jpg",
  "split": "train",
  "label_type": "positive_real",
  "domain": "map_south_china_sea",
  "source_type": "web_screenshot",
  "is_synthetic": false,
  "is_hard_negative": false,
  "has_bbox": true,
  "object_size": "small",
  "occlusion": "partial",
  "degradation": ["jpeg_compression", "low_resolution"]
}
```

Các field nên có:

| Field                 | Ý nghĩa                                                                   |
| --------------------- | ------------------------------------------------------------------------- |
| `label_type`          | positive_real / in_domain_neg / out_domain_neg / hard_neg / synthetic_pos |
| `source_type`         | map / website / document / app_ui / video_frame / poster                  |
| `has_bbox`            | true với positive                                                         |
| `object_size`         | small / medium / large theo COCO hoặc theo area ratio                     |
| `degradation`         | blur / compression / perspective / low-res / occlusion                    |
| `source_id`           | dùng để tránh leakage giữa train-test                                     |
| `synthetic_parent_id` | synthetic ảnh nào sinh từ ảnh gốc nào                                     |

## Bước 3. Chống data leakage

Đây là bước rất quan trọng nếu muốn hướng tới hội nghị tốt hơn.

Không được để:

```text
ảnh gốc nằm trong test
synthetic của ảnh đó nằm trong train
```

Ví dụ sai:

```text
test: map_001.jpg
train: map_001_perspective_aug.jpg
```

Cách đúng:

```text
Nếu ảnh gốc vào test → toàn bộ augmentation/synthetic của nó không được vào train.
```

Nên split theo `source_id` hoặc `parent_id`, không split random theo ảnh.

## Bước 4. Thiết kế split chính

Mình đề xuất split như sau:

```text
Train: 70%
Validation: 10%
Test-Standard: 10%
Test-Robustness: 10%
```

Cụ thể:

### Train

Gồm:

```text
positive_real_train
in_domain_neg_train
out_domain_neg_train
synthetic_pos_train
hard_neg_train
```

### Validation

Dùng để chọn checkpoint và tune threshold:

```text
positive_real_val
in_domain_neg_val
out_domain_neg_val
hard_neg_val nhỏ
```

### Test-Standard

Dùng báo cáo mAP chính:

```text
positive_real_test
negative_mixed_test
```

### Test-Robustness

Dùng báo cáo robustness:

```text
in_domain_neg_test
out_domain_neg_test
hard_neg_test
small_object_test
low_resolution_test
perspective_distorted_test
occluded_test
```

## Bước 5. Thống kê dataset

Bạn cần có bảng và hình thống kê dataset. Bài cũ đã có histogram kích thước ảnh và bbox, chỉ ra ảnh có nhiều độ phân giải và bbox thường nhỏ/mảnh.  Paper mới nên mở rộng thành:

### Bảng thống kê tổng

| Category            | #Images | #Instances | BBox?           | Used in train | Used in test |
| ------------------- | ------: | ---------: | --------------- | ------------- | ------------ |
| Positive real       |    2.2k |        ... | Yes             | Yes           | Yes          |
| In-domain negative  |    2.9k |          0 | No              | Yes           | Yes          |
| Out-domain negative |    2.1k |          0 | No              | Yes           | Yes          |
| Synthetic positive  |     ... |        ... | Yes             | Yes           | No           |
| Hard negative       |     ... |          0 | No / pseudo-box | Yes           | Yes          |

### Hình cần có

1. Distribution of image resolution.
2. Distribution of bbox width/height.
3. Distribution of bbox area ratio.
4. Distribution of object size: small / medium / large.
5. Domain distribution.
6. Example gallery: positive, in-domain negative, out-domain negative, hard negative, synthetic positive.

---

# III. Giai đoạn 2 — Thiết lập baseline chuẩn

Trước khi làm method mới, bạn cần benchmark mạnh. Đây là điều paper cũ còn yếu vì mới so với Faster R-CNN, DETR, SSD. Bài cũ báo cáo DETR ResNet-50 đạt 32.7% mAP, SSD VGG16 đạt 31.0%, nhưng chưa có YOLO/RT-DETR/Cascade R-CNN hiện đại. 

## Bước 1. Chọn mô hình baseline

Tối thiểu nên chạy:

### Nhóm two-stage detector

```text
Faster R-CNN R50-FPN
Faster R-CNN R101-FPN
Cascade R-CNN R50-FPN
```

### Nhóm one-stage detector

```text
YOLOv8s hoặc YOLOv8m
YOLOv9s / YOLOv9m nếu code ổn định
YOLOv11s / YOLOv11m nếu bạn muốn cập nhật hơn
RetinaNet R50-FPN
FCOS R50-FPN
```

### Nhóm transformer detector

```text
RT-DETR-R18/R50
Deformable DETR
DETR baseline nếu muốn so với bài cũ
```

### Nhóm method cũ

```text
Faster R-CNN R50 + DINOv2 KD cũ: L1 + IRM
```

## Bước 2. Chuẩn hóa training protocol

Để so sánh công bằng, cần cố định:

```text
input size
epoch / iterations
optimizer
learning rate schedule
batch size hoặc effective batch size
augmentation
train/val/test split
random seed
confidence threshold khi đo FPR
NMS threshold
```

Mình khuyên:

```text
Seeds: 3 seeds nếu đủ tài nguyên
Epochs: 20 hoặc 30, tùy dataset và model
Save best checkpoint theo validation mAP hoặc validation recall@fixed-FPR
```

Vì moderation cần recall cao, bạn nên chọn checkpoint không chỉ theo mAP mà thêm:

```text
best mAP
best recall at FPR <= x%
```

## Bước 3. Metric chính

Báo cáo object detection metric:

```text
mAP@[0.5:0.95]
AP50
AP75
AR
AP_small
AP_medium
AP_large
AR_small
```

Báo cáo moderation metric:

```text
Image-level Recall
Image-level Precision
F1
False Positive Rate on in-domain negatives
False Positive Rate on out-domain negatives
False Positive Rate on hard negatives
Recall at fixed FPR
FPR at fixed Recall
```

Lý do: mAP không đủ cho bài toán moderation. Một model có mAP cao nhưng báo nhầm nhiều trên bản đồ không vi phạm thì không thực tế.

## Bước 4. Bảng benchmark chính

Bảng này nên là bảng trung tâm của paper:

| Model                 | mAP | AP50 | AP75 | AR | AP_small | FPR-in | FPR-hard | FPS |
| --------------------- | --: | ---: | ---: | -: | -------: | -----: | -------: | --: |
| Faster R-CNN R50-FPN  |     |      |      |    |          |        |          |     |
| Faster R-CNN R101-FPN |     |      |      |    |          |        |          |     |
| Cascade R-CNN         |     |      |      |    |          |        |          |     |
| YOLOv8s               |     |      |      |    |          |        |          |     |
| YOLOv11s              |     |      |      |    |          |        |          |     |
| RT-DETR               |     |      |      |    |          |        |          |     |
| Deformable DETR       |     |      |      |    |          |        |          |     |
| Old KD: L1 + IRM      |     |      |      |    |          |        |          |     |
| Proposed              |     |      |      |    |          |        |          |     |

---

# IV. Giai đoạn 3 — Xây dựng method mới

Method mới nên phát triển trực tiếp từ paper cũ. Bài cũ dùng:

```text
L_KD = β1 · L1 + β2 · Lirm
L_Total = L_rpnc + L_rpnr + L_roic + L_roir + L_KD
```

Trong đó `L1` align embedding predicted region với ground-truth region, còn `Lirm` giữ quan hệ tương quan giữa embedding predicted và ground-truth.  Cấu hình tốt nhất cũ là `β1 = 1.0, β2 = 0.1`. 

Với dataset mới, bạn nên mở rộng thành:

```text
L_total = L_det + β1 L_pos_align + β2 L_rel + β3 L_neg_contrast + β4 L_hn_obj
```

Trong đó:

| Loss             | Mục đích                                                    |
| ---------------- | ----------------------------------------------------------- |
| `L_det`          | Detection loss chuẩn                                        |
| `L_pos_align`    | Align predicted positive ROI với GT positive ROI qua DINOv2 |
| `L_rel`          | Giữ relational structure như IRM                            |
| `L_neg_contrast` | Đẩy hard-negative ROI xa positive prototype                 |
| `L_hn_obj`       | Phạt confidence/objectness cao trên hard-negative ảnh       |

## Bước 1. Làm lại KD cũ thật chắc

Trước khi thêm loss mới, cần tái hiện method cũ trên dataset mới:

```text
Baseline A: Faster R-CNN R50-FPN
Baseline B: Faster R-CNN R50-FPN + L1
Baseline C: Faster R-CNN R50-FPN + L1 + IRM
```

Mục tiêu: xác nhận DINOv2 KD còn có ích khi dataset đã lớn.

## Bước 2. Làm rõ predicted-GT matching

Đây là điểm paper cũ chưa mô tả đủ rõ. Bài mới phải định nghĩa chính xác:

```text
For each GT box g_i:
  select predicted proposals p_j with IoU(p_j, g_i) > τ_pos
  choose top-k by objectness or IoU
  compute DINOv2 embedding for crop(p_j) and crop(g_i)
  apply L_pos_align
```

Đề xuất:

```text
τ_pos = 0.5
top-k = 1 hoặc 3
embedding = DINOv2 crop feature
normalize = L2 normalization
```

Nếu ảnh có nhiều GT box:

```text
match theo IoU cao nhất hoặc Hungarian matching
```

## Bước 3. Thêm hard negative sampling

Trong dataloader, kiểm soát batch composition:

```text
Batch composition đề xuất:
40% positive real/synthetic
30% in-domain negative
20% hard negative
10% out-domain negative
```

Hoặc nếu batch size nhỏ, dùng sampler theo epoch:

```text
Mỗi epoch:
- oversample positive nhỏ
- oversample hard negative
- không để out-domain negative chiếm quá nhiều
```

Mục tiêu là làm model học phân biệt:

```text
nine-dash-line thật
vs
đường cong/đường đứt đoạn/sông/ngòi/đường xá giống nine-dash-line
```

## Bước 4. Thêm negative contrastive distillation

Tạo positive prototype:

```text
c_pos = mean(DINOv2 embeddings of GT positive regions)
```

Với hard negative image, lấy các proposal có confidence cao hoặc objectness cao:

```text
h_j = DINOv2 embedding of hard-negative proposal
```

Dùng loss đẩy xa:

```text
L_neg = max(0, margin - distance(h_j, c_pos))
```

Hoặc dùng cosine:

```text
L_neg = max(0, cosine(h_j, c_pos) - τ)
```

Ý nghĩa:

```text
Nếu một vùng hard-negative trông quá giống nine-dash-line trong embedding space,
model bị phạt để giảm nhầm lẫn.
```

## Bước 5. Thêm hard-negative objectness penalty

Với ảnh hard negative không có object thật, nếu model dự đoán box với confidence cao:

```text
L_hn_obj = mean(max(0, score_j - τ_score))
```

Ví dụ:

```text
τ_score = 0.3 hoặc 0.5
chỉ lấy top-N predictions trên hard-negative images
```

Loss này trực tiếp giảm false positive.

## Bước 6. Đặt tên method

Bạn nên đặt tên method rõ ràng. Ví dụ:

```text
HN-DINO-KD: Hard Negative-Aware DINOv2 Knowledge Distillation
```

hoặc:

```text
DACD: Domain-Aware Contrastive Distillation
```

Tên method nên phản ánh điểm mới:

```text
hard-negative-aware
domain-aware
contrastive
distillation
thin object detection
```

---

# V. Giai đoạn 4 — Thực nghiệm chính

## Experiment 1 — Dataset scale effect

Mục tiêu: chứng minh dataset 11k có giá trị hơn dataset cũ.

Chạy cùng một model trên các subset:

| Training data   | mAP | AP_small | AR | FPR-in | FPR-hard |
| --------------- | --: | -------: | -: | -----: | -------: |
| 836 old dataset |     |          |    |        |          |
| 2k subset       |     |          |    |        |          |
| 5k subset       |     |          |    |        |          |
| 11k full        |     |          |    |        |          |

Kết luận kỳ vọng:

```text
Tăng data giúp mAP/recall tăng,
nhưng nếu không có hard negative thì FPR-hard vẫn cao.
```

## Experiment 2 — Negative domain effect

Mục tiêu: chứng minh in-domain negative và hard negative quan trọng.

Ablation dataset:

| Training setting       | Pos | In-neg | Out-neg | Hard-neg | Synthetic | mAP | FPR-in | FPR-hard |
| ---------------------- | --- | ------ | ------- | -------- | --------- | --: | -----: | -------: |
| Pos only               | ✓   | ✗      | ✗       | ✗        | ✗         |     |        |          |
| Pos + out-neg          | ✓   | ✗      | ✓       | ✗        | ✗         |     |        |          |
| Pos + in-neg           | ✓   | ✓      | ✗       | ✗        | ✗         |     |        |          |
| Pos + in/out-neg       | ✓   | ✓      | ✓       | ✗        | ✗         |     |        |          |
| + hard-neg             | ✓   | ✓      | ✓       | ✓        | ✗         |     |        |          |
| + hard-neg + synthetic | ✓   | ✓      | ✓       | ✓        | ✓         |     |        |          |

Kết luận cần đạt:

```text
In-domain negative giảm false positive trên bản đồ biển đảo.
Hard negative giảm nhầm sông/ngòi/đường xá.
Synthetic positive tăng recall với biến dạng phối cảnh.
```

## Experiment 3 — Main comparison

So sánh proposed với detector hiện đại:

| Model                | mAP | AP50 | AP75 | AP_small | AR | FPR-in | FPR-hard | FPS |
| -------------------- | --: | ---: | ---: | -------: | -: | -----: | -------: | --: |
| Faster R-CNN R50-FPN |     |      |      |          |    |        |          |     |
| Cascade R-CNN        |     |      |      |          |    |        |          |     |
| YOLOv8s              |     |      |      |          |    |        |          |     |
| YOLOv11s             |     |      |      |          |    |        |          |     |
| RT-DETR              |     |      |      |          |    |        |          |     |
| Old DINOv2 KD        |     |      |      |          |    |        |          |     |
| Proposed HN-DINO-KD  |     |      |      |          |    |        |          |     |

Kết luận mong muốn:

```text
Proposed không nhất thiết nhanh nhất,
nhưng có trade-off tốt nhất giữa recall, AP_small và FPR-hard.
```

## Experiment 4 — KD loss ablation

Dựa trên loss cũ và loss mới:

| Method          | L1 | IRM | Neg contrast | HN objectness | mAP | AP_small | FPR-hard |
| --------------- | -- | --- | ------------ | ------------- | --: | -------: | -------: |
| Detector only   | ✗  | ✗   | ✗            | ✗             |     |          |          |
| + L1            | ✓  | ✗   | ✗            | ✗             |     |          |          |
| + L1 + IRM      | ✓  | ✓   | ✗            | ✗             |     |          |          |
| + Neg contrast  | ✓  | ✓   | ✓            | ✗             |     |          |          |
| + HN objectness | ✓  | ✓   | ✗            | ✓             |     |          |          |
| Full proposed   | ✓  | ✓   | ✓            | ✓             |     |          |          |

Kỳ vọng:

```text
L1/IRM tăng mAP và AP_small.
Neg contrast và HN objectness giảm FPR-hard.
Full proposed cân bằng tốt nhất.
```

## Experiment 5 — Hyperparameter ablation

Bài cũ đã có bảng β1/β2 và cho thấy `β1 = 1.0, β2 = 0.1` tốt nhất.  Bài mới nên mở rộng:

### Với β3 — negative contrast

|   β3 | mAP | AP_small | FPR-hard |
| ---: | --: | -------: | -------: |
|  0.0 |     |          |          |
| 0.05 |     |          |          |
|  0.1 |     |          |          |
|  0.5 |     |          |          |
|  1.0 |     |          |          |

### Với hard-negative ratio

| Hard-neg ratio in batch | mAP | Recall | FPR-hard |
| ----------------------: | --: | -----: | -------: |
|                      0% |     |        |          |
|                     10% |     |        |          |
|                     20% |     |        |          |
|                     30% |     |        |          |
|                     40% |     |        |          |

Kỳ vọng: hard-neg ratio quá cao có thể giảm recall positive, nên cần tìm điểm cân bằng.

## Experiment 6 — Synthetic positive ablation

| Synthetic augmentation    | mAP | Recall-perspective | Recall-small | FPR-hard |
| ------------------------- | --: | -----------------: | -----------: | -------: |
| None                      |     |                    |              |          |
| Rotation/scale only       |     |                    |              |          |
| + perspective             |     |                    |              |          |
| + blur/compression        |     |                    |              |          |
| + full synthetic pipeline |     |                    |              |          |

Kết luận cần chứng minh:

```text
Perspective synthetic tăng robustness với ảnh bị nghiêng, chụp màn hình, tài liệu scan.
Blur/compression tăng robustness với ảnh chất lượng thấp.
```

## Experiment 7 — Teacher ablation

Nếu còn thời gian, so sánh teacher:

| Teacher encoder    | mAP | AP_small | FPR-hard |
| ------------------ | --: | -------: | -------: |
| None               |     |          |          |
| DINOv2-S           |     |          |          |
| DINOv2-B           |     |          |          |
| CLIP ViT-B/16      |     |          |          |
| MAE/ViT pretrained |     |          |          |

Mục tiêu: chứng minh DINOv2 không phải lựa chọn ngẫu nhiên.

## Experiment 8 — Input resolution / small object ablation

Nine-dash line thường nhỏ/mảnh, nên input size rất quan trọng.

| Input size | mAP | AP_small | FPS | GPU memory |
| ---------: | --: | -------: | --: | ---------: |
|        640 |     |          |     |            |
|        800 |     |          |     |            |
|       1024 |     |          |     |            |
|       1280 |     |          |     |            |

Kết luận có thể là:

```text
Resolution cao tăng AP_small nhưng giảm FPS.
Proposed method giữ hiệu quả tốt ở resolution vừa phải.
```

---

# VI. Giai đoạn 5 — Robustness evaluation

Đây là phần giúp paper khác biệt so với bài cũ.

## Test 1. In-domain negative robustness

Input:

```text
bản đồ biển đảo không có nine-dash line
```

Metric:

```text
FPR-in-domain = số ảnh bị báo nhầm / tổng ảnh in-domain negative
```

Bảng:

| Model        | FPR-in-domain ↓ | Avg false boxes/image ↓ | Max confidence on negatives ↓ |
| ------------ | --------------: | ----------------------: | ----------------------------: |
| Faster R-CNN |                 |                         |                               |
| YOLO         |                 |                         |                               |
| Old KD       |                 |                         |                               |
| Proposed     |                 |                         |                               |

## Test 2. Hard-negative robustness

Input:

```text
sông ngòi
đường xá
đường biên
contour line
dashed line
coastline
```

Metric:

```text
FPR-hard
Avg confidence of false positives
False positive boxes per image
```

Có thể chia hard negative thành subcategory:

| Hard negative type      | FPR Faster R-CNN | FPR Old KD | FPR Proposed |
| ----------------------- | ---------------: | ---------: | -----------: |
| River                   |                  |            |              |
| Road                    |                  |            |              |
| Coastline               |                  |            |              |
| Administrative boundary |                  |            |              |
| Dashed lines            |                  |            |              |

## Test 3. Degradation robustness

Tạo test biến đổi từ positive test real:

```text
blur
JPEG compression
low resolution
brightness shift
perspective transform
partial crop
occlusion
```

Bảng:

| Corruption  | Severity | Recall baseline | Recall proposed |
| ----------- | -------: | --------------: | --------------: |
| JPEG        |    1/3/5 |                 |                 |
| Blur        |    1/3/5 |                 |                 |
| Perspective |    1/3/5 |                 |                 |
| Low-res     |    1/3/5 |                 |                 |

## Test 4. Threshold analysis

Vì moderation cần đặt threshold, bạn nên vẽ:

```text
Precision-Recall curve
ROC-like curve for image-level decision
Recall vs FPR
FPR-hard vs confidence threshold
```

Báo cáo:

```text
Recall at FPR-in <= 1%
Recall at FPR-hard <= 1%
FPR-hard at Recall >= 90%
```

---

# VII. Giai đoạn 6 — Error analysis

Phần này cần làm rất kỹ. Một paper tốt không chỉ báo cáo số mà còn giải thích lỗi.

## Bước 1. Gom lỗi thành 4 nhóm

### False negative

Model bỏ sót positive.

Phân loại nguyên nhân:

```text
FN-small: object quá nhỏ
FN-blur: ảnh mờ/nén
FN-occlusion: bị che khuất
FN-crop: chỉ thấy một phần nine-dash line
FN-context: background phức tạp
FN-perspective: bị nghiêng/biến dạng
```

### False positive

Model báo nhầm trên negative.

Phân loại:

```text
FP-river
FP-road
FP-coastline
FP-border
FP-dashed-line
FP-map-grid
FP-text/watermark
```

### Localization error

Model phát hiện đúng vùng nhưng bbox lệch.

```text
IoU < 0.5 nhưng box nằm gần object
box quá rộng, chứa nhiều background
box chỉ bao một phần object
```

### Duplicate detection

Một object bị detect nhiều box.

```text
NMS chưa tối ưu
confidence threshold chưa hợp lý
```

## Bước 2. Tạo bảng error taxonomy

| Error type | Count | Percentage | Main cause          | Possible fix           |
| ---------- | ----: | ---------: | ------------------- | ---------------------- |
| FN-small   |       |            | object rất nhỏ      | higher resolution/FPN  |
| FN-blur    |       |            | compression         | degradation aug        |
| FP-river   |       |            | curved thin line    | hard negative contrast |
| FP-road    |       |            | dashed road pattern | hard negative sampling |
| Loc error  |       |            | bbox loose          | box loss/segmentation  |

## Bước 3. Làm figure qualitative

Nên có một figure 4×4:

```text
Row 1: baseline miss, proposed detects
Row 2: baseline FP on river/road, proposed rejects
Row 3: both fail cases
Row 4: challenging success: perspective/blur/small object
```

Mỗi ảnh nên có:

```text
GT box: green
baseline prediction: red
proposed prediction: blue
confidence score
```

## Bước 4. Phân tích embedding

Nếu dùng DINOv2 contrastive, nên có UMAP/t-SNE:

```text
positive ROI embeddings
hard-negative ROI embeddings
in-domain negative ROI embeddings
```

So sánh:

```text
Before contrastive loss: hard negative gần positive
After contrastive loss: hard negative tách xa positive
```

Đây là bằng chứng rất mạnh cho method.

---

# VIII. Giai đoạn 7 — Viết paper theo cấu trúc mới

## Section 1 — Introduction

Nêu vấn đề:

```text
Nine-dash line thường nhỏ, mảnh, biến dạng, xuất hiện trong bản đồ/tài liệu/screenshot.
Thách thức chính không chỉ là detect positive mà còn tránh false positive trên hard negative giống hình thái.
```

Nêu gap của bài cũ:

```text
small dataset
limited negative diversity
limited robustness evaluation
```

Nêu đóng góp mới:

```text
large-scale domain-aware dataset
hard-negative-aware distillation
comprehensive robustness benchmark
```

## Section 2 — Dataset

Mô tả dataset 11k:

```text
collection
annotation
taxonomy
split
statistics
quality control
leakage prevention
```

Nên có bảng taxonomy + histogram + example gallery.

## Section 3 — Method

Mô tả:

```text
Base detector
DINOv2 region embedding
positive alignment loss
relational loss
hard negative contrastive loss
hard negative objectness penalty
training objective
```

Công thức:

```text
L_total = L_det + β1 L_pos + β2 L_rel + β3 L_neg + β4 L_hn
```

## Section 4 — Experiments

Gồm:

```text
implementation details
baseline models
metrics
main comparison
domain robustness
ablation studies
```

## Section 5 — Error Analysis

Gồm:

```text
false positive taxonomy
false negative taxonomy
qualitative cases
embedding visualization
failure discussion
```

## Section 6 — Conclusion

Nhấn mạnh:

```text
robustness
hard negative
deployment for moderation
future work: segmentation/mask/open-world monitoring
```

---

# IX. Timeline thực hiện đề xuất

## Tuần 1 — Dataset audit

Việc cần làm:

```text
chuẩn hóa annotation COCO format
gán metadata
kiểm tra duplicate/source leakage
tạo split train/val/test/robustness
vẽ thống kê dataset
```

Output:

```text
dataset_v1_coco.json
metadata.csv
split files
dataset statistics figures
```

## Tuần 2 — Baseline training

Train:

```text
Faster R-CNN R50-FPN
YOLOv8s hoặc YOLOv11s
RT-DETR
Old DINOv2 KD
```

Output:

```text
main benchmark sơ bộ
best checkpoint
log training
PR curves
```

## Tuần 3 — Hard negative experiments

Làm:

```text
hard negative sampling
FPR-hard evaluation
negative domain ablation
synthetic positive ablation
```

Output:

```text
dataset ablation table
domain robustness table
```

## Tuần 4 — Method mới

Implement:

```text
negative contrastive loss
hard negative objectness penalty
positive prototype
proposal mining trên hard negative
```

Output:

```text
proposed method checkpoint
loss ablation
hyperparameter sweep β3/β4
```

## Tuần 5 — Robustness và error analysis

Làm:

```text
corruption robustness
threshold analysis
false positive taxonomy
false negative taxonomy
qualitative visualization
t-SNE/UMAP embedding
```

Output:

```text
error analysis section
figures
failure cases
```

## Tuần 6 — Viết paper

Làm:

```text
viết dataset section
method section
experiment section
chốt tables/figures
soạn abstract/introduction/conclusion
```

Output:

```text
paper draft v1
supplementary material nếu cần
```

---

# X. Checklist thực nghiệm tối thiểu

Nếu tài nguyên hạn chế, đây là bộ tối thiểu nên làm:

```text
[ ] Dataset statistics đầy đủ
[ ] Split không leakage
[ ] Faster R-CNN R50-FPN baseline
[ ] YOLOv8/YOLOv11 baseline
[ ] RT-DETR baseline
[ ] Old DINOv2 KD baseline
[ ] Proposed hard-negative-aware KD
[ ] Dataset ablation: pos / in-neg / out-neg / hard-neg / synthetic
[ ] Loss ablation: detector / L1 / L1+IRM / +neg contrast / full
[ ] Metrics: mAP, AP50, AP75, AR, AP_small, FPR-in, FPR-hard
[ ] Error analysis: FP/FN taxonomy
[ ] Qualitative figure
```

Nếu làm được thêm:

```text
[ ] 3 random seeds
[ ] external test set
[ ] teacher ablation
[ ] resolution ablation
[ ] robustness under blur/compression/perspective
[ ] UMAP/t-SNE embedding visualization
```

---

# XI. Kết luận chiến lược

Kế hoạch tốt nhất là biến bài mới thành một nghiên cứu về:

```text
Robust small/thin object detection under domain-specific hard negatives
```

Dataset 11k của bạn có giá trị lớn nhất ở **negative structure**, không chỉ ở số lượng ảnh. Vì vậy các kết quả quan trọng nhất không chỉ là:

```text
mAP tăng bao nhiêu?
```

mà phải là:

```text
FPR trên bản đồ biển đảo không vi phạm giảm bao nhiêu?
FPR trên sông/ngòi/đường xá giống nine-dash line giảm bao nhiêu?
Recall trên object nhỏ/mờ/biến dạng tăng bao nhiêu?
Synthetic positive có giúp robustness không?
DINOv2 distillation có tách được hard negative khỏi positive trong embedding space không?
```

Nếu trả lời được các câu hỏi đó bằng bảng thực nghiệm, ablation và error analysis rõ ràng, bài mới sẽ mạnh hơn RIVF rất nhiều và có narrative đủ tốt để hướng tới hội nghị xếp hạng cao hơn.
