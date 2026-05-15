Dataset audit
Tìm ảnh thiếu annotation, kiểm tra bbox hợp lệ, chuẩn hóa custom JSON sang COCO, thêm split, source_id, synthetic_parent_id.

Split chống leakage
Không random theo ảnh. Với positive_synthetic, phải biết nó sinh từ ảnh gốc nào. Nếu ảnh gốc vào val/test thì synthetic cùng parent không được ở train.

Baseline tối thiểu
Chạy ít thôi trước:
Faster R-CNN R50-FPN, YOLOv8/YOLOv11, RT-DETR, old DINOv2 KD.
Đừng mở rộng Cascade, Deformable DETR, teacher ablation, 3 seeds ngay từ đầu.

Metric moderation
Định nghĩa rõ image-level FPR:
ảnh negative bị false positive nếu có ít nhất 1 box score >= threshold.
Threshold phải chọn bằng validation, không tune trên test.

Method mới
Cẩn thận với L_neg_contrast: nếu chỉ tính trên embedding DINOv2 frozen thì loss không thật sự cập nhật detector. Loss phải gắn với feature/proposal/head của student hoặc score/objectness của detector. L_hn_obj thì hợp lý hơn và dễ chứng minh tác động trực tiếp lên FPR-hard.

Ablation tối thiểu cho paper
Chỉ cần đủ mạnh:
dataset ablation, loss ablation, FPR-in/FPR-hard, AP_small, qualitative FP/FN.