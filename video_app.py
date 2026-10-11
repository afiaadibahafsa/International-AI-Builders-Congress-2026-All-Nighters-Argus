"""
video_app.py  —  Conveyor package detector + XAI + Groq Vision
==========================================================
Behaviours:
  • Normal video shown by default — NO heatmap overlay
  • When a box centre reaches the green detection line:
      - Heatmap activates on the video for that box only
      - XAI panel updates with EigenCAM + Groq Vision result
      - Heatmap fades back to normal video after 2 seconds
  • Groq Vision (Llama 3.2) analyses flap AND damage
  • Report written to package_data_YYYY-MM-DD.txt

Controls:  Q/ESC=quit  P=pause  H=force heatmap on/off  S=save frame
Run:       python video_app.py
Install:   pip install groq opencv-python ultralytics torch
"""

import argparse
import base64
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH          = "best.pt"
VIDEO_PATH          = "co.mp4"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
DETECTION_LINE_FRAC = 0.41
LINE_OFFSET         = 18        # ± pixels tolerance
HEATMAP_SHOW_SECS   = 2.5       # seconds heatmap stays visible after crossing
REPORT_FOLDER       = "cropped_packages"
ALERT_SOUND         = "alert.mp3"

GROQ_API_KEY        = "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"        # ← paste your Groq API key (console.groq.com)
# Hardcoded fallback vision model IDs — used only if live discovery fails.
# The code auto-discovers all current vision models from the Groq API first.
# If you get "not available" errors, check: console.groq.com/docs/models
GROQ_MODELS_FALLBACK = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3.6-27b",
]

FONT       = cv2.FONT_HERSHEY_SIMPLEX
BOX_COLOR  = (255, 0, 255)
LINE_COLOR = (0, 255, 0)
CROSS_COLOR= (0, 220, 220)      # yellow-green when box is crossing
ALPHA      = 0.55
FILE_LOCK  = threading.Lock()


# ── EigenCAM hook ─────────────────────────────────────────────────────────────

class TrackHook:
    """
    Hooks into yolo.track() forward pass to capture backbone activations.
    Attach before track(), detach after, then call get_heatmap().
    """
    def __init__(self, pt_model: torch.nn.Module):
        self._acts   = None
        self._handle = None
        self._layer  = None
        self._lock   = threading.Lock()

        all_convs = [(n, m) for n, m in pt_model.named_modules()
                     if isinstance(m, torch.nn.Conv2d)]
        if not all_convs:
            print("[EigenCAM] No Conv2d found"); return

        idx = max(0, int(len(all_convs) * 0.35))
        for i in range(idx, -1, -1):
            if all_convs[i][1].kernel_size[0] > 1:
                idx = i; break

        self._layer = all_convs[idx][1]
        print(f"[EigenCAM] Hook → {all_convs[idx][0]} "
              f"kernel={self._layer.kernel_size} "
              f"out_ch={self._layer.out_channels}")

    def attach(self):
        if self._layer is None: return
        with self._lock: self._acts = None
        def _hook(m, inp, out):
            with self._lock:
                if self._acts is None:
                    self._acts = out.detach().cpu().float()
        self._handle = self._layer.register_forward_hook(_hook)

    def detach(self):
        if self._handle is not None:
            self._handle.remove(); self._handle = None

    def get_heatmap(self, out_h: int, out_w: int,
                    box=None) -> np.ndarray:
        with self._lock: acts = self._acts
        if acts is None:
            return np.zeros((out_h, out_w), np.float32)

        feat = acts.squeeze(0)
        C, Hf, Wf = feat.shape
        flat = feat.reshape(C, Hf * Wf)
        cam  = flat.abs().mean(0).reshape(Hf, Wf).numpy().astype(np.float32)

        try:
            mat = flat.numpy().astype(np.float32)
            mat = mat - mat.mean(axis=1, keepdims=True)
            cov = mat @ mat.T
            v   = np.ones(C, np.float32)
            for _ in range(30):
                v   = cov @ v
                nrm = float(np.linalg.norm(v))
                if nrm < 1e-8: break
                v  /= nrm
            pca = (v @ mat).reshape(Hf, Wf)
            pca = np.maximum(pca, 0)
            if pca.max() > 0:
                cam = pca / pca.max()
        except Exception:
            pass

        cam = np.maximum(cam, 0)
        if cam.max() > 0: cam /= cam.max()

        hm = cv2.resize(cam, (out_w, out_h))

        # Suppress outside the box region so attention shows ON box only
        if box is not None:
            x1, y1, x2, y2 = box
            mask = np.zeros_like(hm)
            mask[y1:y2, x1:x2] = 1.0
            hm = hm * mask

        return hm


def blend_heatmap(frame_bgr: np.ndarray,
                  heatmap: np.ndarray,
                  alpha: float = ALPHA) -> np.ndarray:
    hm8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    col = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    return cv2.addWeighted(col, alpha, frame_bgr, 1 - alpha, 0)


# ── Gemini ─────────────────────────────────────────────────────────────────────

_VISION_PROMPT = (
    "You are a strict quality-control inspector on a conveyor belt. "
    "Examine this image of a cardboard package very carefully.\n\n"
    "Question 1 — FLAP OPEN:\n"
    "Is the box top/lid/flap open, unsealed, or raised? "
    "Even a partially open or bent-up flap counts as YES.\n\n"
    "Question 2 — PHYSICAL DAMAGE:\n"
    "Look very carefully at the cardboard for ANY of these signs:\n"
    "- Bent, folded, or creased cardboard walls or corners\n"
    "- Crushed, dented, or compressed sides\n"
    "- Torn, ripped, or punctured cardboard\n"
    "- Flaps that are misaligned, warped, or bent out of shape\n"
    "- Any visible deformation from the box original flat shape\n"
    "- Structural weakness: box looks like it has been squeezed or dropped\n"
    "IMPORTANT: Be sensitive — even minor bending or creasing counts as YES. "
    "If the box does not look perfectly flat and square on all visible sides, "
    "answer YES. Only answer NO if the box looks completely undamaged, "
    "perfectly flat and square with no deformation whatsoever.\n\n"
    "Reply in EXACTLY this format (one word per line):\n"
    "FLAP: Yes\n"
    "DAMAGE: No"
)


def _parse_result(text: str):
    """Parse FLAP/DAMAGE lines from model response."""
    flap = damage = "Unknown"
    for line in text.strip().lower().splitlines():
        line = line.strip()
        if line.startswith("flap"):
            flap   = "Yes" if "yes" in line else "No"
        elif line.startswith("damage"):
            damage = "Yes" if "yes" in line else "No"
    return flap, damage


# Cache of discovered vision model IDs (populated on first call)
_groq_vision_models_cache = []


def _discover_groq_vision_models():
    """
    Calls the Groq /openai/v1/models endpoint to get the live model list,
    then filters to vision-capable ones. Caches the result.
    Returns a list of model ID strings.
    """
    global _groq_vision_models_cache
    if _groq_vision_models_cache:
        return _groq_vision_models_cache

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        models  = client.models.list()
        # Vision models on Groq typically have "vision" in id OR
        # are known multimodal families (llama-4, qwen-vl, etc.)
        vision_keywords = ["vision", "llama-4", "maverick", "scout",
                           "qwen3", "qwen-vl", "pixtral"]
        found = []
        for m in models.data:
            mid = m.id.lower()
            if any(kw in mid for kw in vision_keywords):
                found.append(m.id)
        if found:
            print(f"[Groq] Discovered {len(found)} vision model(s): {found}")
            _groq_vision_models_cache = found
            return found
    except Exception as e:
        print(f"[Groq] Model discovery failed: {e}")

    # Fall back to hardcoded list
    print(f"[Groq] Using fallback model list: {GROQ_MODELS_FALLBACK}")
    _groq_vision_models_cache = GROQ_MODELS_FALLBACK
    return GROQ_MODELS_FALLBACK


def gemini_analyse(crop_bgr: np.ndarray):
    """
    Analyses flap + damage using Groq vision API.
    Auto-discovers current vision models from the live Groq model list
    so it never breaks when models are deprecated.
    Returns (flap, damage) each "Yes" / "No" / "Unknown".
    """
    if not GROQ_API_KEY:
        return "Unknown", "Unknown"

    try:
        from groq import Groq
    except ImportError:
        print("[Groq] Package not installed. Run: pip install groq")
        return "Unknown", "Unknown"

    ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return "Unknown", "Unknown"

    b64      = base64.b64encode(buf.tobytes()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64}"
    client   = Groq(api_key=GROQ_API_KEY)
    models   = _discover_groq_vision_models()

    for model_name in models:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": data_uri}},
                            {"type": "text", "text": _VISION_PROMPT},
                        ],
                    }],
                    max_tokens=64,
                    temperature=0,
                )
                text = response.choices[0].message.content
                flap, damage = _parse_result(text)
                if flap != "Unknown" or damage != "Unknown":
                    if attempt > 0 or model_name != models[0]:
                        print(f"[Groq] Success with {model_name} "
                              f"(attempt {attempt+1})")
                    return flap, damage

            except Exception as e:
                err = str(e)
                if "rate" in err.lower() or "429" in err or "503" in err:
                    wait = 3 * (attempt + 1)
                    print(f"[Groq] {model_name} rate-limited "
                          f"(attempt {attempt+1}/3), retrying in {wait}s...")
                    time.sleep(wait)
                elif any(k in err.lower() for k in
                         ["404","not found","decommission","deprecated",
                          "invalid_request","not available"]):
                    print(f"[Groq] {model_name} unavailable, trying next...")
                    # Remove from cache so we don't try it again this session
                    if model_name in _groq_vision_models_cache:
                        _groq_vision_models_cache.remove(model_name)
                    break
                else:
                    print(f"[Groq] {e}")
                    return "Unknown", "Unknown"

    print("[Groq] All discovered models failed.")
    return "Unknown", "Unknown"


# ── Local fallback ────────────────────────────────────────────────────────────

def local_analyse(crop_bgr: np.ndarray):
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 10:
        return "No", "No"
    gray  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    blur  = cv2.GaussianBlur(clahe.apply(gray), (5,5), 0)
    edges = cv2.Canny(blur, 40, 120)
    top   = edges[:h//2, :]
    lines = cv2.HoughLinesP(top,1,np.pi/180,30,
                             minLineLength=w//3,maxLineGap=10)
    hl = 0
    if lines is not None:
        for seg in lines.reshape(-1,4):
            x1,y1,x2,y2=int(seg[0]),int(seg[1]),int(seg[2]),int(seg[3])
            if abs(np.degrees(np.arctan2(y2-y1,x2-x1)))<20: hl+=1
    row_vars = np.var(gray[:max(1,h//4),:].astype(np.float32),axis=1)
    bright   = bool(np.any(row_vars>400))
    ratio    = float(edges[:h//2,:].mean())/max(float(edges[h//2:,:].mean()),1.)
    flap     = "Yes" if sum([hl>=2, bright, ratio>1.4])>=2 else "No"
    return flap, "No"


# ── XAI side panel ────────────────────────────────────────────────────────────

def make_panel(heatmap, crop_bgr, label, conf,
               flap, damage, analysing, pkg_count,
               pw=340, ph=600):
    p = np.full((ph, pw, 3), 25, np.uint8)

    cv2.rectangle(p,(0,0),(pw,42),(40,40,80),-1)
    cv2.putText(p,"XAI Explanation",(8,29),
                FONT,0.68,(255,255,120),2,cv2.LINE_AA)
    y = 50

    # EigenCAM thumbnail
    cv2.putText(p,"EigenCAM  (model attention)",(6,y+13),
                FONT,0.40,(160,200,255),1,cv2.LINE_AA); y+=17
    th,tw = 158,pw-12
    hm_t  = cv2.resize(heatmap,(tw,th))
    hm8   = np.uint8(255*np.clip(hm_t,0,1))
    p[y:y+th,6:6+tw] = cv2.applyColorMap(hm8,cv2.COLORMAP_JET)
    for i in range(tw):
        v=int(255*i/max(tw-1,1))
        c=cv2.applyColorMap(np.array([[[v]]],np.uint8),cv2.COLORMAP_JET)[0,0]
        p[y+th-7:y+th-1,6+i]=c
    cv2.putText(p,"low",(6,y+th+13),FONT,0.35,(140,140,140),1)
    cv2.putText(p,"high",(tw-18,y+th+13),FONT,0.35,(140,140,140),1)
    y+=th+18

    cv2.line(p,(4,y),(pw-4,y),(60,60,60),1); y+=7

    # Crop thumbnail
    cv2.putText(p,"Package at detection line:",(6,y+13),
                FONT,0.40,(160,200,255),1,cv2.LINE_AA); y+=17
    ch_max=108
    if crop_bgr is not None and crop_bgr.size>0:
        ch,cw=crop_bgr.shape[:2]
        sc=min((pw-12)/max(cw,1),ch_max/max(ch,1))
        nw2=max(1,int(cw*sc)); nh2=max(1,int(ch*sc))
        cr=cv2.resize(crop_bgr,(nw2,nh2))
        hm_c=cv2.resize(heatmap,(nw2,nh2))
        hm_c8=np.uint8(255*np.clip(hm_c,0,1))
        cr_ov=cv2.addWeighted(
            cv2.applyColorMap(hm_c8,cv2.COLORMAP_JET),0.40,cr,0.60,0)
        p[y:y+nh2,6:6+nw2]=cr_ov
    else:
        cv2.rectangle(p,(6,y),(pw-6,y+ch_max),(50,50,50),-1)
        cv2.putText(p,"waiting for box at green line...",
                    (8,y+ch_max//2),FONT,0.33,(120,120,120),1,cv2.LINE_AA)
    y+=ch_max+8

    cv2.line(p,(4,y),(pw-4,y),(60,60,60),1); y+=7

    cv2.putText(p,"Analysis Result:",(6,y+13),
                FONT,0.50,(255,255,100),1,cv2.LINE_AA); y+=23

    def row(k,v,vc):
        nonlocal y
        cv2.putText(p,k,(10,y),FONT,0.45,(180,180,180),1,cv2.LINE_AA)
        cv2.putText(p,v,(90,y),FONT,0.50,vc,1,cv2.LINE_AA)
        y+=23

    row("Label :",label,(200,255,200))
    row("Conf  :",f"{conf:.1%}",(200,255,200))

    if analysing:
        row("Flap  :","Analysing...",(160,160,60))
        row("Damage:","Analysing...",(160,160,60))
    else:
        row("Flap  :",flap,
            (60,80,240) if flap=="Yes" else (80,220,80))
        row("Damage:",damage,
            (60,80,240) if damage=="Yes" else (80,220,80))

    src="via Groq AI" if GROQ_API_KEY else "via local analysis"
    if not analysing:
        cv2.putText(p,src,(10,y),FONT,0.36,(120,160,255),1,cv2.LINE_AA)
        y+=18

    verdict = "CHECKING..." if analysing else \
              "ANOMALY" if (flap=="Yes" or damage=="Yes") else "NORMAL"
    vc2=(80,80,20) if verdict=="CHECKING..." else \
        (30,30,180) if verdict=="ANOMALY" else (20,130,20)
    cv2.rectangle(p,(6,y+2),(pw-6,y+32),vc2,-1)
    cv2.putText(p,verdict,(pw//2-50,y+22),
                FONT,0.62,(255,255,255),2,cv2.LINE_AA)
    y+=40

    cv2.line(p,(4,y),(pw-4,y),(60,60,60),1); y+=6
    cv2.putText(p,f"Total packages analysed: {pkg_count}",
                (6,y+14),FONT,0.43,(180,220,180),1,cv2.LINE_AA)
    return p


# ── Sound ─────────────────────────────────────────────────────────────────────

class Sound:
    def __init__(self):
        self._ok=False; self._on=False
        if not os.path.exists(ALERT_SOUND):
            print(f"[Sound] '{ALERT_SOUND}' not found — silent."); return
        try:
            import pygame; pygame.mixer.init()
            pygame.mixer.music.load(ALERT_SOUND); self._ok=True
        except Exception as e:
            print(f"[Sound] {e} — silent.")
    def play(self):
        if self._ok and not self._on:
            import pygame; pygame.mixer.music.play(-1); self._on=True
    def stop(self):
        if self._ok and self._on:
            import pygame; pygame.mixer.music.stop(); self._on=False


# ── App ───────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, source, save_path=None):
        print(f"Loading {MODEL_PATH} on {DEVICE} ...")
        self.yolo     = YOLO(MODEL_PATH)
        self.names    = self.yolo.names
        self.pt_model = self.yolo.model
        self.pt_model.eval()
        print(f"Classes: {self.names}")

        self.hook = TrackHook(self.pt_model)

        if GROQ_API_KEY:
            print(f"Groq key found — vision models will be auto-discovered from API.")
        else:
            print("WARNING: No GROQ_API_KEY — local analysis only.")

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            sys.exit(f"[ERROR] Cannot open: {source!r}")
        self.source  = source
        self.fps_src = self.cap.get(cv2.CAP_PROP_FPS) or 30

        os.makedirs(REPORT_FOLDER, exist_ok=True)
        self.report = f"package_data_{time.strftime('%Y-%m-%d')}.txt"
        if not os.path.exists(self.report):
            with open(self.report,"w",encoding="utf-8") as f:
                f.write("Timestamp | Track ID | Package Type"
                        " | Flap Open | Damage\n")
                f.write("-"*80+"\n")

        self.crossed_ids = set()
        self.sound = Sound()
        self.writer = None
        if save_path:
            fourcc=cv2.VideoWriter_fourcc(*"XVID")
            self.writer=cv2.VideoWriter(
                save_path,fourcc,self.fps_src,(1360,600))
            print(f"Saving to: {save_path}")

        self._id_results   = {}
        self._results_lock = threading.Lock()

        self._heatmap    = np.zeros((600,1020),np.float32)
        self._hmap_lock  = threading.Lock()

        # Timestamp of last line crossing — controls heatmap visibility
        self._last_cross_time = 0.0

    def _bg_analyse(self, crop_bgr, heatmap_crop, tid, conf, label):
        ts=time.strftime("%Y-%m-%d_%H-%M-%S")
        cv2.imwrite(os.path.join(REPORT_FOLDER,f"pkg_{tid}_{ts}.jpg"),crop_bgr)

        if GROQ_API_KEY:
            flap,damage=gemini_analyse(crop_bgr)
        else:
            flap,damage=local_analyse(crop_bgr)

        with self._results_lock:
            if tid in self._id_results:
                self._id_results[tid].update({
                    "flap":flap,"damage":damage,"analysing":False})

        with FILE_LOCK:
            with open(self.report,"a",encoding="utf-8") as f:
                f.write(f"{ts} | Track ID: {tid} | box | {flap} | {damage}\n")

        tag=[]
        if flap=="Yes":   tag.append("FLAP OPEN")
        if damage=="Yes": tag.append("DAMAGED")
        print(f"  [ID {tid:>3}] {label} {conf:.0%} | "
              f"Flap:{flap} Damage:{damage}"
              +(f"  *** {'|'.join(tag)} ***" if tag else "  OK"))

        if flap=="Yes" or damage=="Yes": self.sound.play()
        else:                            self.sound.stop()

    def run(self):
        force_hm=False      # H key toggles forced heatmap mode
        paused=False; fc=0; sc=0
        t_prev=time.time(); fps_d=0.0; frame=None

        panel_label="waiting for box..."; panel_conf=0.0
        panel_flap="\u2014"; panel_damage="\u2014"
        panel_crop=None; panel_heatmap=None; panel_analysing=False

        WIN="Package Detection [Q=quit P=pause H=heatmap S=save]"
        cv2.namedWindow(WIN,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN,1360,600)
        print("\nRunning — Q=quit  P=pause  H=force-heatmap  S=save\n")

        while True:
            if not paused:
                ret,raw=self.cap.read()
                if not ret:
                    if isinstance(self.source,str):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
                    break

                frame=cv2.resize(raw,(1020,600))
                fc+=1

                self.hook.attach()
                results=self.yolo.track(frame,persist=True,verbose=False)
                self.hook.detach()

                line_x=int(1020*DETECTION_LINE_FRAC)
                now=time.time()

                if results and results[0].boxes is not None:
                    r    =results[0]
                    boxes=r.boxes.xyxy.cpu().numpy().astype(int)
                    ids  =(r.boxes.id.cpu().numpy().astype(int)
                           .flatten().tolist()
                           if r.boxes.id is not None
                           else list(range(len(boxes))))
                    confs=r.boxes.conf.cpu().numpy().tolist()
                    clss =r.boxes.cls.cpu().numpy().astype(int).tolist()

                    for (x1,y1,x2,y2),tid,conf,cls in \
                            zip(boxes,ids,confs,clss):
                        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                        tid=int(tid)
                        label=self.names.get(int(cls),str(cls))
                        cx=(x1+x2)//2

                        crossing=(line_x-LINE_OFFSET<cx<line_x+LINE_OFFSET)
                        bcolor=CROSS_COLOR if crossing else BOX_COLOR
                        cv2.rectangle(frame,(x1,y1),(x2,y2),bcolor,2)
                        cv2.putText(frame,
                                    f"{label} {conf:.0%} ID:{tid}",
                                    (x1,max(y1-6,12)),
                                    FONT,0.52,bcolor,1,cv2.LINE_AA)

                        if crossing and tid not in self.crossed_ids:
                            self.crossed_ids.add(tid)
                            self._last_cross_time=now  # start heatmap timer

                            y1c=max(0,y1); y2c=min(600,y2)
                            x1c=max(0,x1); x2c=min(1020,x2)
                            if y2c>y1c and x2c>x1c:
                                crop=frame[y1c:y2c,x1c:x2c].copy()
                                hm_full=self.hook.get_heatmap(
                                    600,1020,box=(x1c,y1c,x2c,y2c))
                                hm_crop=hm_full[y1c:y2c,x1c:x2c].copy()

                                with self._results_lock:
                                    self._id_results[tid]={
                                        "label":label,"conf":conf,
                                        "flap":"\u2014","damage":"\u2014",
                                        "crop":crop.copy(),
                                        "heatmap":hm_crop.copy(),
                                        "analysing":True,
                                    }

                                panel_label    =label
                                panel_conf     =conf
                                panel_crop     =crop.copy()
                                panel_heatmap  =hm_crop.copy()
                                panel_flap     ="\u2014"
                                panel_damage   ="\u2014"
                                panel_analysing=True

                                with self._hmap_lock:
                                    self._heatmap=hm_full.copy()

                                threading.Thread(
                                    target=self._bg_analyse,
                                    args=(crop,hm_crop,tid,conf,label),
                                    daemon=True).start()

                # Update panel from last-crossed result
                if self.crossed_ids:
                    last_tid=max(self.crossed_ids)
                    with self._results_lock:
                        res=self._id_results.get(last_tid)
                    if res is not None:
                        panel_label    =res["label"]
                        panel_conf     =res["conf"]
                        panel_flap     =res["flap"]
                        panel_damage   =res["damage"]
                        panel_crop     =res["crop"]
                        panel_heatmap  =res.get("heatmap")
                        panel_analysing=res["analysing"]

                t_now=time.time()
                fps_d=1.0/max(t_now-t_prev,1e-6)
                t_prev=t_now

            if frame is None:
                continue

            # ── Decide whether to show heatmap ────────────────────────────────
            # Show heatmap for HEATMAP_SHOW_SECS after each line crossing,
            # OR when H key forces it on.
            elapsed_since_cross=time.time()-self._last_cross_time
            heatmap_active=(force_hm or
                            elapsed_since_cross < HEATMAP_SHOW_SECS)

            disp=frame.copy()
            if heatmap_active:
                with self._hmap_lock:
                    hm=self._heatmap.copy()
                # Smooth fade: alpha decreases as time since crossing grows
                if not force_hm and elapsed_since_cross < HEATMAP_SHOW_SECS:
                    fade=1.0-elapsed_since_cross/HEATMAP_SHOW_SECS
                    alpha_now=ALPHA*fade
                    if alpha_now>0.05:
                        disp=blend_heatmap(disp,hm,alpha=alpha_now)
                else:
                    with self._hmap_lock:
                        hm=self._heatmap.copy()
                    disp=blend_heatmap(disp,hm)

            # Detection line
            line_x=int(1020*DETECTION_LINE_FRAC)
            cv2.line(disp,(line_x,0),(line_x,600),LINE_COLOR,2)
            cv2.putText(disp,"Detection Line",(line_x+4,18),
                        FONT,0.44,LINE_COLOR,1,cv2.LINE_AA)

            # Status bar
            hm_status="[HEATMAP:FORCED]" if force_hm else \
                       f"[HEATMAP:{max(0,HEATMAP_SHOW_SECS-elapsed_since_cross):.1f}s]" \
                       if heatmap_active else "[NORMAL VIEW]"
            cv2.rectangle(disp,(0,562),(1020,600),(20,20,20),-1)
            cv2.putText(disp,
                        f"FPS:{fps_d:.1f}"
                        f"  Analysed:{len(self.crossed_ids)}"
                        f"  {hm_status}"
                        f"  {'[PAUSED]' if paused else ''}",
                        (6,588),FONT,0.40,(200,200,200),1,cv2.LINE_AA)

            # XAI panel — use box-focused heatmap
            if panel_heatmap is not None and panel_heatmap.size>0:
                # Resize to full frame for thumbnail display
                xai_hm=np.zeros((600,1020),np.float32)
                # Place box heatmap back in original position using panel crop
                # For the thumbnail we just resize the crop heatmap directly
                xai_hm_thumb=panel_heatmap
            else:
                with self._hmap_lock:
                    xai_hm_thumb=self._heatmap.copy()

            xai=make_panel(
                xai_hm_thumb,panel_crop,
                panel_label,panel_conf,
                panel_flap,panel_damage,
                panel_analysing,len(self.crossed_ids))

            combined=np.hstack([disp,xai])
            if self.writer: self.writer.write(combined)
            cv2.imshow(WIN,combined)

            key=cv2.waitKey(1)&0xFF
            if key in (ord("q"),ord("Q"),27): break
            elif key in (ord("p"),ord("P")):
                paused=not paused
                print("Paused." if paused else "Resumed.")
            elif key in (ord("h"),ord("H")):
                force_hm=not force_hm
                print(f"Force heatmap:{'ON' if force_hm else 'OFF'}")
            elif key in (ord("s"),ord("S")):
                fn=f"saved_frame_{sc:03d}.png"
                cv2.imwrite(fn,combined)
                print(f"Saved:{fn}"); sc+=1

        self.cap.release()
        if self.writer: self.writer.release()
        cv2.destroyAllWindows()
        self.sound.stop()
        print(f"\nDone. {fc} frames. Report:{self.report}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default=VIDEO_PATH)
    p.add_argument("--save",  default=None)
    args=p.parse_args()
    src=args.source
    try: src=int(src)
    except: pass
    App(src,args.save).run()

if __name__=="__main__":
    main()