import os
import time
import numpy as np
import onnxruntime as ort
import psutil

# ---------------- Paths ----------------
data_root = r"C:\Users\MAINAK\ps14\data"
test_path = os.path.join(data_root, "cache", "test.npz")
model_path = os.path.join(data_root, "models", "model_qat_int8_packed.onnx")

CLASSES = ["Forest", "Cropland", "Urban", "Water", "Cloud", "CloudShadow", "Ship"]
EVAL_SOURCE = {
    "Forest": "eurosat", "Cropland": "eurosat", "Urban": "eurosat", "Water": "eurosat",
    "Cloud": "cloudsen12", "CloudShadow": "cloudsen12", "Ship": "hrsid"
}

# ---------------- Memory Tracker ----------------
process = psutil.Process(os.getpid())

def get_ram_mb():
    # Returns the physical memory (RSS - Resident Set Size) used by this process in MB
    return process.memory_info().rss / (1024 * 1024)

# ---------------- Metric Helpers ----------------
def prep(xb):
    return np.ascontiguousarray(xb.transpose(0, 3, 1, 2)).astype(np.float32) / 255.0

def pr_metrics(y, s):
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    total = max(float(y.sum()), 1.0)
    rec = tp / total
    ap = float((prec * y).sum() / total)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return ap, float(f1.max())

def evaluate(probs, Y, M, source):
    per_class = {}
    for j, c in enumerate(CLASSES):
        sel = (M[:, j] == 1) & (source == EVAL_SOURCE[c])
        y, s = Y[sel, j], probs[sel, j]
        if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
            per_class[c] = (float("nan"), float("nan"))
        else:
            per_class[c] = pr_metrics(y, s)
    return per_class, float(np.nanmean([v[0] for v in per_class.values()]))

# ---------------- Execution ----------------
print(f"Loading test dataset from {test_path}...")
test_data = np.load(test_path)
X, Y, M, source = test_data["X"], test_data["Y"], test_data["M"], test_data["source"]
print(f"Test tiles loaded: {len(X)}")

# 1. Check Baseline RAM (Data loaded, model NOT loaded)
ram_baseline = get_ram_mb()

print(f"\nLoading packed 8-bit model: {model_path}...")
so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

# 2. Check RAM after model is loaded into memory
ram_model_loaded = get_ram_mb()

BATCH_SIZE = 512
out = []
peak_inference_ram = ram_model_loaded

print(f"Running ONNX inference & tracking memory...")
t0 = time.time()

for i in range(0, len(X), BATCH_SIZE):
    batch_x = prep(X[i:i + BATCH_SIZE])
    out.append(sess.run(None, {input_name: batch_x})[0])
    
    # 3. Track peak RAM spikes during active processing
    current_ram = get_ram_mb()
    if current_ram > peak_inference_ram:
        peak_inference_ram = current_ram

print(f"Inference finished in {time.time() - t0:.2f} seconds.")

logits = np.concatenate(out).astype(np.float64)
probs = 1.0 / (1.0 + np.exp(-logits))
class_metrics, macro_ap = evaluate(probs, Y, M, source)

# ---------------- Report ----------------
print("\n" + "="*65)
print(f"{'Class':<15} | {'Average Precision (AP)':<22} | {'Max F1 Score':<15}")
print("="*65)
for c in CLASSES:
    ap, f1 = class_metrics.get(c, (float('nan'), float('nan')))
    print(f"{c:<15} | {ap:<22.4f} | {f1:<15.4f}")
print("-" * 65)
print(f"{'MACRO AP':<15} | {macro_ap:<22.4f} |")
print("="*65)

print("\n" + "="*65)
print(f"{'MEMORY PROFILING (RAM)':<65}")
print("="*65)
print(f"Baseline RAM (Dataset Loaded): {ram_baseline:.2f} MB")
print(f"RAM After ONNX Model Load:     {ram_model_loaded:.2f} MB")
print(f"Model Working Set Overhead:    {ram_model_loaded - ram_baseline:.2f} MB")
print(f"Peak RAM During Inference:     {peak_inference_ram:.2f} MB")
print(f"Max Inference Spike:           {peak_inference_ram - ram_model_loaded:.2f} MB")
print("="*65)
