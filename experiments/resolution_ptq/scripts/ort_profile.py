"""Where does DS-Net spend time on the Pi? ORT node profiling, aggregated by SegFormer block and op type."""
import collections, glob, json, os, re, sys
import numpy as np, onnxruntime as ort
model, threads, runs, skip = sys.argv[1], int(sys.argv[2]), 25, 5
so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.enable_profiling = True; so.profile_file_prefix = "/tmp/ortprof_" + os.path.basename(model)[:-5]
sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
x = np.random.randn(*[d if isinstance(d, int) else 1 for d in inp.shape]).astype(np.float32)
for _ in range(runs): sess.run(None, {inp.name: x})
ev = json.load(open(sess.end_profiling()))
win = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") == "Session" and e.get("name") == "model_run")[skip:]
def in_win(t): return any(a <= t <= b for a, b in win)
nodes = [e for e in ev if e.get("cat") == "Node" and e["name"].endswith("_kernel_time") and in_win(e["ts"])]
def block(n):
    if m := re.search(r"patch_embeddings\.(\d)", n): return f"enc stage{int(m[1])+1} patch-embed"
    if m := re.search(r"block\.(\d)\.\d+/(attention|mlp|layer_norm_\d)", n):
        return f"enc stage{int(m[1])+1} {'attention' if m[2] == 'attention' else ('mix-ffn' if m[2] == 'mlp' else 'block LN')}"
    if m := re.search(r"encoder/layer_norm\.(\d)", n): return f"enc stage{int(m[1])+1} final LN"
    if "decode_head/linear_c" in n: return "dec per-stage MLP + upsample"
    if re.search(r"decode_head/(linear_fuse|batch_norm|activation)", n): return "dec fuse (1024->256 @1/4)"
    if "decode_head/classifier" in n: return "dec classifier"
    if "decode_head" in n: return "dec other (concat etc.)"
    return "final upsample / other"
by_b, by_op = collections.Counter(), collections.Counter()
for e in nodes:
    by_b[block(e["name"])] += e["dur"]; by_op[e["args"].get("op_name", "?")] += e["dur"]
tot = sum(by_b.values()); per_run = tot / len(win) / 1000
print(f"{os.path.basename(model)}  threads={threads}  node time per run {per_run:.1f} ms (runs={len(win)})")
grp = collections.Counter()
for k, v in by_b.items(): grp["ENCODER" if k.startswith("enc") else ("DECODER" if k.startswith("dec") else "OTHER")] += v
print("  " + "  ".join(f"{k} {v/tot*100:.0f}%" for k, v in grp.most_common()))
for k, v in by_b.most_common(10): print(f"   {v/tot*100:5.1f}%  {v/len(win)/1000:6.1f} ms  {k}")
print("  top ops: " + ", ".join(f"{k} {v/tot*100:.0f}%" for k, v in by_op.most_common(8)))
oth = collections.Counter()
for e in nodes:
    if block(e["name"]) == "final upsample / other": oth[e["args"].get("op_name", "?")] += e["dur"]
print("  inside 'final upsample / other': " + ", ".join(f"{k} {v/tot*100:.1f}%" for k, v in oth.most_common(6)))
