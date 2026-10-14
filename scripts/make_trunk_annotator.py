"""Build a self-contained HTML page for manually marking the visible trunk position.

Independent validation (MMA3001 brief: "manual inspection supported by defined
criteria"). For each test tree the page shows the lower part of the summer image
with a horizontal guide line at row 230 of 256 (just above the ground). The
annotator clicks the **centre of the visible trunk where it crosses the line**.
If the trunk is not visible at the line, press "Not visible".

Criterion, fixed before looking at any model output: *the horizontal centre of
the bark visible at row 230; if the trunk is leaning, the centre where it
crosses the line; ignore posts and stakes.*

The page saves ``trunk_annotations.csv`` (tree_id, x_px at 256 px, visible flag),
which ``scripts/trunk_check.py`` compares against the hand labels and the models.

    python scripts/make_trunk_annotator.py   # -> doc/trunk_annotator.html
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treeid.splits import split_by_row  # noqa: E402

Y_LINE = 230   # row of the guide line in the 256 x 256 frame
Y0 = 120       # first row shown
SCALE = 2      # display magnification


def main():
    d = dict(np.load("data/cache_256.npz"))
    _, _, te = split_by_row(d["row"])
    rng = np.random.default_rng(7)          # shuffled so the annotator cannot follow row order
    idx = rng.permutation(np.flatnonzero(te))
    items = []
    for i in idx:
        rgb = (d["x"][i][:3].astype(np.float32).transpose(1, 2, 0) * 255).astype(np.uint8)[Y0:]
        img = cv2.resize(rgb, (256 * SCALE, (256 - Y0) * SCALE), interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
        items.append({"id": str(d["tree"][i]), "src": "data:image/jpeg;base64," + base64.b64encode(buf).decode()})
    html = TEMPLATE.replace("__ITEMS__", json.dumps(items)).replace("__YLINE__", str((Y_LINE - Y0) * SCALE)) \
                   .replace("__SCALE__", str(SCALE))
    out = Path("doc/trunk_annotator.html")
    out.write_text(html)
    print(f"wrote {out} with {len(items)} trees ({out.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Trunk annotator</title>
<style>
body{font:15px system-ui,sans-serif;margin:0;padding:16px;background:#f6f6f4;color:#222}
.wrap{max-width:820px;margin:auto}
#stage{position:relative;display:inline-block;cursor:crosshair;border:1px solid #bbb}
#stage img{display:block}
#line{position:absolute;left:0;right:0;height:0;border-top:2px dashed #ffeb3b;pointer-events:none}
#mark{position:absolute;width:2px;background:#e53935;top:0;bottom:0;display:none;pointer-events:none}
button{font:inherit;padding:6px 14px;margin:6px 6px 0 0;cursor:pointer}
.meta{margin:8px 0;color:#555}
</style></head><body><div class="wrap">
<h2>Visible trunk position: test trees</h2>
<p>Click the <b>centre of the visible trunk where it crosses the dashed line</b>. Ignore posts and stakes.
If no trunk is visible at the line, press <i>Not visible</i>. Keys: <b>N</b> = not visible, <b>Backspace</b> = back.</p>
<div class="meta" id="meta"></div>
<div id="stage"><img id="img" alt="tree"><div id="line"></div><div id="mark"></div></div><br>
<button id="nv">Not visible</button><button id="back">Back</button><button id="save">Download CSV</button>
<div class="meta" id="done"></div></div>
<script>
const items=__ITEMS__, YL=__YLINE__, S=__SCALE__; let k=0; const ans={};
const img=document.getElementById('img'), line=document.getElementById('line'), mark=document.getElementById('mark');
line.style.top=YL+'px';
function show(){ if(k>=items.length){document.getElementById('meta').textContent='All '+items.length+' done. Download the CSV.';return;}
  img.src=items[k].src; const a=ans[items[k].id];
  mark.style.display=(a&&a.v)?'block':'none'; if(a&&a.v) mark.style.left=(a.x*S)+'px';
  document.getElementById('meta').textContent=(k+1)+' / '+items.length+'  ·  '+items[k].id;
  document.getElementById('done').textContent=Object.keys(ans).length+' answered'; }
function next(){k=Math.min(k+1,items.length); show();}
document.getElementById('stage').addEventListener('click',e=>{ if(k>=items.length)return;
  const r=img.getBoundingClientRect(); const x=(e.clientX-r.left)*(img.naturalWidth/r.width)/S;
  ans[items[k].id]={x:Math.round(x*10)/10,v:1}; next();});
document.getElementById('nv').onclick=()=>{ if(k<items.length){ans[items[k].id]={x:'',v:0}; next();}};
document.getElementById('back').onclick=()=>{k=Math.max(0,k-1); show();};
document.addEventListener('keydown',e=>{ if(e.key==='n'||e.key==='N')document.getElementById('nv').click();
  if(e.key==='Backspace'){e.preventDefault();document.getElementById('back').click();}});
document.getElementById('save').onclick=()=>{ let csv='tree_id,x_px_256,visible\n';
  for(const it of items){const a=ans[it.id]; if(a) csv+=it.id+','+a.x+','+a.v+'\n';}
  const b=new Blob([csv],{type:'text/csv'}); const u=URL.createObjectURL(b); const l=document.createElement('a');
  l.href=u; l.download='trunk_annotations.csv'; l.click(); URL.revokeObjectURL(u);};
show();
</script></body></html>"""


if __name__ == "__main__":
    main()
