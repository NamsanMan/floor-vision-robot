"""라벨 진행 상황: split마다 reviewed 체크된 장수와, 아직 안 한 첫 사진(labelme 파일 목록에서 몇 번째인지).
labelme 검색창은 쓰지 말 것(걸러진 목록에서 D/A가 멈추고, 지운 뒤 체크하면 labelme가 닫힘) → 목록을 스크롤해서 그 번째 사진을 클릭."""
import glob
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
for split in ("train", "test"):
    files = sorted(glob.glob(os.path.join(here, "ft150", "labels", split, "*.json")))
    flags = [bool((json.load(open(f, encoding="utf-8")).get("flags") or {}).get("reviewed")) for f in files]
    todo = [k for k, done in enumerate(flags) if not done]
    line = f"{split}: {sum(flags)}/{len(files)} reviewed"
    if todo:
        k = todo[0]
        line += f"  -> 다음: {os.path.splitext(os.path.basename(files[k]))[0]} (목록 {k + 1}번째)"
    else:
        line += "  (끝)"
    print(line)
