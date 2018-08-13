import  numpy as np

with open("/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/eval_dir/AP_Result/ap_results.txt", "r") as f:
    data = f.readlines()

aps = []
for li in data:
    ap = li.strip().split(", ")[-1]
    if ap == "AP":continue
    aps.append(float(ap))
aps = np.array(aps)
print np.mean(aps)
