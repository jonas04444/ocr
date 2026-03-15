#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   🎥 Lecteur de Plaques Belges — Vidéo / Webcam             ║
║   Détection améliorée : faible lumière / voitures garées    ║
║                                                              ║
║   Webcam : python plaque_video.py --webcam                  ║
║   MP4    : python plaque_video.py --video fichier.mp4       ║
║                                                              ║
║   Touches :  ESPACE → pause/reprise   Q → quitter           ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, re, os, argparse, time, threading, queue
from datetime import datetime

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    print("[ERREUR] pip install pytesseract pillow opencv-python")
    sys.exit(1)

RS = "\033[0m"; B = "\033[1m"; CYAN = "\033[96m"; GRN = "\033[92m"
YLW = "\033[93m"; RED = "\033[91m"; GRY = "\033[90m"

PATTERN_NOUVELLE = re.compile(r'^[0-9]-[A-Z]{3}-[0-9]{3}$')
PATTERN_ANCIENNE = re.compile(r'^[A-Z]{1,3}-[0-9]{1,4}-[A-Z]{0,3}$')

TESS_CONFIGS = [
    r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
    r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
]

CLR_VALIDE   = (50,  220,  50)
CLR_CANDIDAT = (0,   165, 255)
CLR_OVERLAY  = (20,   20,  20)
CLR_PAUSE    = (0,   200, 255)


def ameliorer_luminosite(img_bgr):
    """CLAHE sur le canal V pour corriger la faible luminosite."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    v_eq  = clahe.apply(v)
    return cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2BGR)


def _iou(a, b):
    x1 = max(a['x'], b['x']); y1 = max(a['y'], b['y'])
    x2 = min(a['x']+a['w'], b['x']+b['w']); y2 = min(a['y']+a['h'], b['y']+b['h'])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0: return 0.0
    return inter / (a['w']*a['h'] + b['w']*b['h'] - inter)


def _fusionner_candidates(candidates):
    gardes = []
    for cand in sorted(candidates, key=lambda c: -c['score']):
        if not any(_iou(cand, g) > 0.4 for g in gardes):
            gardes.append(cand)
    return gardes


def detecter_zone_plaque(img_bgr):
    h_img, w_img = img_bgr.shape[:2]
    candidates   = []
    img_eq = ameliorer_luminosite(img_bgr)
    hsv    = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)

    # Blanc large (faible lumiere), blanc strict, jaune
    masques = [
        ("blanc_large",  cv2.inRange(hsv, (0,0,120), (180,55,255)),  0.15),
        ("blanc_strict", cv2.inRange(hsv, (0,0,180), (180,40,255)),  0.10),
        ("jaune",        cv2.inRange(hsv, (15,40,80), (40,255,255)), 0.10),
    ]
    masques = [(n, cv2.inRange(hsv, np.array(lo), np.array(hi)), sa)
               for n, (lo, hi), sa in [
                   ("blanc_large",  ([0,0,120],[180,55,255]),  0.15),
                   ("blanc_strict", ([0,0,180],[180,40,255]),  0.10),
                   ("jaune",        ([15,40,80],[40,255,255]), 0.10),
               ]]

    for nom, masque, seuil_aire in masques:
        k_f = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 6))
        k_o = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 4))
        masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, k_f)
        masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN,  k_o)
        for c in cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) < 700: continue
            _, (rw, rh), _ = cv2.minAreaRect(c)
            if rw < rh: rw, rh = rh, rw
            if rh < 4:  continue
            sr = 1.0 - abs(rw/rh - 4.6) / 4.6
            if sr < 0.25: continue
            sa = min(cv2.contourArea(c) / (w_img * h_img * seuil_aire), 1.0)
            x, y, w, hh = cv2.boundingRect(c)
            candidates.append({'x':x,'y':y,'w':w,'h':hh,'score':sr*0.65+sa*0.35,'methode':nom})

    # Gradient (robuste faible lumiere)
    gris = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    gris = cv2.fastNlMeansDenoising(gris, h=12, templateWindowSize=7, searchWindowSize=15)
    gris = cv2.GaussianBlur(gris, (5,5), 0)
    grad = cv2.magnitude(cv2.Sobel(gris,cv2.CV_64F,1,0,ksize=3),
                         cv2.Sobel(gris,cv2.CV_64F,0,1,ksize=3))
    mx = grad.max()
    if mx > 0:
        grad = np.uint8(np.clip(grad/mx*255, 0, 255))
        _, seuil = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        seuil = cv2.morphologyEx(seuil, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT,(30,4)))
        for c in cv2.findContours(seuil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) < 400: continue
            x, y, w, hh = cv2.boundingRect(c)
            if hh == 0: continue
            sr = 1.0 - abs(w/hh - 4.6) / 4.6
            if sr < 0.20: continue
            candidates.append({'x':x,'y':y,'w':w,'h':hh,'score':sr*0.5,'methode':'gradient'})

    # Canny rectangles (fallback)
    gris2 = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    gris2 = cv2.bilateralFilter(gris2, 11, 75, 75)
    bords = cv2.Canny(gris2, 30, 120)
    bords = cv2.morphologyEx(bords, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT,(5,2)))
    for c in cv2.findContours(bords, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) < 300: continue
        poly = cv2.approxPolyDP(c, 0.04*cv2.arcLength(c,True), True)
        if len(poly) not in (4,5,6): continue
        x, y, w, hh = cv2.boundingRect(c)
        if hh == 0: continue
        sr = 1.0 - abs(w/hh - 4.6) / 4.6
        if sr < 0.25: continue
        roi_lum = gris2[y:y+hh, x:x+w]
        if roi_lum.size > 0 and roi_lum.mean() < 60: continue
        candidates.append({'x':x,'y':y,'w':w,'h':hh,'score':sr*0.45,'methode':'canny'})

    candidates = _fusionner_candidates(candidates)
    candidates.sort(key=lambda c: -c['score'])
    return candidates[:6]


def _ordonner_points(pts):
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
    rect[1]=pts[np.argmin(diff)]; rect[3]=pts[np.argmax(diff)]
    return rect


def corriger_perspective(img_bgr, x, y, w, h, marge=8):
    x1=max(0,x-marge); y1=max(0,y-marge)
    x2=min(img_bgr.shape[1],x+w+marge); y2=min(img_bgr.shape[0],y+h+marge)
    roi = img_bgr[y1:y2, x1:x2]
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return roi
    c_max = max(contours, key=cv2.contourArea)
    poly  = cv2.approxPolyDP(c_max, 0.02*cv2.arcLength(c_max,True), True)
    if len(poly) == 4:
        pts = poly.reshape(4,2).astype(np.float32)
        pts_ord = _ordonner_points(pts)
        tl,tr,br,bl = pts_ord
        larg = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
        haut = int(max(np.linalg.norm(tl-bl), np.linalg.norm(tr-br)))
        if larg>0 and haut>0:
            dst = np.array([[0,0],[larg-1,0],[larg-1,haut-1],[0,haut-1]], dtype=np.float32)
            roi = cv2.warpPerspective(roi, cv2.getPerspectiveTransform(pts_ord,dst),(larg,haut))
    return roi


def pretraiter_plaque(roi_bgr):
    """
    Plaques belges : texte ROUGE sur fond BLANC.
    Canal VERT = rouge tres sombre, blanc tres clair -> contraste maximal.
    """
    h, w = roi_bgr.shape[:2]
    if h == 0 or w == 0: return {}
    roi_bgr = cv2.resize(roi_bgr, (w*3, h*3), interpolation=cv2.INTER_CUBIC)

    # V1 : canal vert - optimal rouge sur blanc
    cg = roi_bgr[:, :, 1]
    cg = cv2.fastNlMeansDenoising(cg, h=8, templateWindowSize=7, searchWindowSize=15)
    _, v1 = cv2.threshold(cg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # V2 : canal vert inverse
    v2 = cv2.bitwise_not(v1)

    # V3 : gris standard + CLAHE
    gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.fastNlMeansDenoising(gris, h=10, templateWindowSize=7, searchWindowSize=21)
    gc   = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4,4)).apply(gris)
    _, v3 = cv2.threshold(gc, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # V4 : adaptatif sur canal vert
    v4 = cv2.adaptiveThreshold(cg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 9)

    # V5 : sharpen
    blur = cv2.GaussianBlur(gc, (0,0), 2)
    _, v5 = cv2.threshold(cv2.addWeighted(gc,1.5,blur,-0.5,0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return {'vert':v1, 'vert_inv':v2, 'otsu':v3, 'adaptatif':v4, 'sharp':v5}


def _forcer_chiffre(c):
    M={'O':'0','Q':'0','D':'0','I':'1','L':'1','l':'1','Z':'2','S':'5','G':'6','B':'8','g':'9','A':'4'}
    return M.get(c,c) if not c.isdigit() else c

def _forcer_lettre(c):
    M={'0':'O','1':'I','5':'S','6':'G','8':'B','4':'A'}
    return M.get(c,c) if not c.isalpha() else c

def corriger_caracteres(texte):
    texte = re.sub(r'[\s_]', '-', texte).strip('-')
    # Supprimer les tirets multiples
    texte = re.sub(r'-{2,}', '-', texte)
    # Enlever les espaces autour des tirets
    texte = re.sub(r'\s*-\s*', '-', texte)

    # Tenter de reconstruire le format si on a 7 chars sans tirets
    if len(texte) == 7 and '-' not in texte:
        texte = f"{texte[0]}-{texte[1:4]}-{texte[4:7]}"

    if len(texte)==9 and texte[1]=='-' and texte[5]=='-':
        chars = list(texte)
        chars[0] = _forcer_chiffre(chars[0])
        for i in (2,3,4): chars[i] = _forcer_lettre(chars[i])
        for i in (6,7,8): chars[i] = _forcer_chiffre(chars[i])
        # Corriger un tiret final parasite ex: 1-BEM-24-
        if chars[8] == '-':
            chars[8] = '4'  # confusion tiret/4 courante
        return ''.join(chars)
    return texte

def scorer_plaque(texte):
    if not texte: return 0.0
    if PATTERN_NOUVELLE.match(texte): return 1.0
    if PATTERN_ANCIENNE.match(texte): return 0.85
    score = 0.0
    if len(texte)==9: score+=0.3
    elif len(texte)==7: score+=0.2
    if len(texte)>1 and texte[1]=='-': score+=0.2
    if len(texte)>5 and texte[5]=='-': score+=0.2
    if re.search(r'\d',texte): score+=0.1
    if re.search(r'[A-Z]',texte): score+=0.1
    return min(score,0.99)

def est_valide(texte):
    return bool(PATTERN_NOUVELLE.match(texte) or PATTERN_ANCIENNE.match(texte))

def ocr_plaque(variantes):
    resultats = {}
    for _, img in variantes.items():
        img_m = cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,value=255)
        for cfg in TESS_CONFIGS:
            try:
                texte = re.sub(r'[^A-Z0-9\-]','',
                               pytesseract.image_to_string(img_m,config=cfg,lang='eng').strip().upper())
                if not texte: continue
                texte = corriger_caracteres(texte)
                score = scorer_plaque(texte)
                if texte not in resultats or score > resultats[texte]['score']:
                    resultats[texte] = {'texte':texte,'score':score,'valide':est_valide(texte)}
            except: continue
    return sorted(resultats.values(), key=lambda r: -r['score'])

# Mémoire des zones valides (persistance entre frames)
_zones_memoire = []
_MEMOIRE_MAX_AGE = 15


def _mettre_a_jour_memoire(nouvelles_candidates):
    global _zones_memoire
    for z in _zones_memoire:
        z['age'] += 1
    for cand in nouvelles_candidates:
        trouve = False
        for z in _zones_memoire:
            if _iou(cand, z) > 0.3:
                alpha = 0.4
                z['x'] = int(alpha*cand['x'] + (1-alpha)*z['x'])
                z['y'] = int(alpha*cand['y'] + (1-alpha)*z['y'])
                z['w'] = int(alpha*cand['w'] + (1-alpha)*z['w'])
                z['h'] = int(alpha*cand['h'] + (1-alpha)*z['h'])
                z['score'] = max(z['score'], cand['score'])
                z['age'] = 0
                trouve = True
                break
        if not trouve:
            _zones_memoire.append({**cand, 'age': 0})
    _zones_memoire[:] = [z for z in _zones_memoire if z['age'] < _MEMOIRE_MAX_AGE]


def analyser_frame(frame):
    global _zones_memoire
    nouvelles = detecter_zone_plaque(frame)
    _mettre_a_jour_memoire(nouvelles)
    zones_a_analyser = list(_zones_memoire) + [
        c for c in nouvelles if not any(_iou(c,z)>0.3 for z in _zones_memoire)
    ]
    detections = []
    for cand in zones_a_analyser[:6]:
        roi = corriger_perspective(frame, cand['x'], cand['y'], cand['w'], cand['h'])
        res = ocr_plaque(pretraiter_plaque(roi))
        if res and res[0]['score'] >= 0.5:
            detections.append({**res[0],'x':cand['x'],'y':cand['y'],'w':cand['w'],'h':cand['h']})
    return detections


def dessiner_frame(frame, detections, etat):
    out = frame.copy()
    hf, wf = out.shape[:2]
    for det in detections:
        x,y,w,hh = det['x'],det['y'],det['w'],det['h']
        col = CLR_VALIDE if det['valide'] else CLR_CANDIDAT
        cv2.rectangle(out,(x,y),(x+w,y+hh),col,3 if det['valide'] else 2)
        tc = min(w,hh)//4
        for (px,py),(dx,dy) in [((x,y),(1,1)),((x+w,y),(-1,1)),((x,y+hh),(1,-1)),((x+w,y+hh),(-1,-1))]:
            cv2.line(out,(px,py),(px+dx*tc,py),col,4)
            cv2.line(out,(px,py),(px,py+dy*tc),col,4)
        label = f"{det['texte']}  {det['score']:.0%}"
        (lw,lh),bl = cv2.getTextSize(label,cv2.FONT_HERSHEY_DUPLEX,0.9,2)
        ly = max(y-12,lh+8)
        ov = out.copy()
        cv2.rectangle(ov,(x,ly-lh-8),(x+lw+12,ly+bl+2),col,-1)
        cv2.addWeighted(ov,0.75,out,0.25,0,out)
        cv2.putText(out,label,(x+6,ly-2),cv2.FONT_HERSHEY_DUPLEX,0.9,(10,10,10),2)

    hud = 52
    ov2 = out.copy()
    cv2.rectangle(ov2,(0,hf-hud),(wf,hf),CLR_OVERLAY,-1)
    cv2.addWeighted(ov2,0.7,out,0.3,0,out)
    cv2.putText(out,f"SOURCE: {etat['source']}",(12,hf-hud+20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(180,180,180),1)
    cv2.putText(out,f"PLAQUES: {etat['nb_valides']}",(12,hf-hud+40),cv2.FONT_HERSHEY_SIMPLEX,0.55,(180,180,180),1)
    if etat['derniere_valide']:
        txt=f"DERNIERE: {etat['derniere_valide']['texte']}"
        (tw,_),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_DUPLEX,0.75,2)
        cv2.putText(out,txt,(wf//2-tw//2,hf-10),cv2.FONT_HERSHEY_DUPLEX,0.75,CLR_VALIDE,2)
    if etat['pause']:
        ptxt="|| PAUSE [ESPACE pour reprendre]"
        (pw,_),_=cv2.getTextSize(ptxt,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
        cv2.putText(out,ptxt,(wf-pw-12,hf-hud+20),cv2.FONT_HERSHEY_SIMPLEX,0.65,CLR_PAUSE,2)
    else:
        rtxt="ESPACE=pause | Q=quitter"
        (rw,_),_=cv2.getTextSize(rtxt,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        cv2.putText(out,rtxt,(wf-rw-12,hf-hud+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(130,130,130),1)
    cv2.putText(out,datetime.now().strftime("%H:%M:%S"),(wf-90,hf-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),1)
    return out


class WorkerOCR(threading.Thread):
    def __init__(self,q_in,q_out):
        super().__init__(daemon=True); self.q_in=q_in; self.q_out=q_out; self.actif=True
    def run(self):
        while self.actif:
            try: frame=self.q_in.get(timeout=0.5)
            except queue.Empty: continue
            try:
                dets=analyser_frame(frame)
                while not self.q_out.empty():
                    try: self.q_out.get_nowait()
                    except queue.Empty: break
                self.q_out.put(dets)
            except: pass
    def arreter(self): self.actif=False


def boucle_video(cap, nom_source, args):
    q_in=queue.Queue(maxsize=2); q_out=queue.Queue(maxsize=2)
    worker=WorkerOCR(q_in,q_out); worker.start()
    etat={'source':nom_source,'pause':False,'nb_valides':0,'derniere_valide':None,'plaques_vues':set()}
    detections_courantes=[]; frame_pause=None; derniere_envoi=0
    fenetre="Plaques Belges - Detection renforcee"
    print(f"\n{CYAN}{B}Lecture demarree : {nom_source}{RS}")
    print(f"{GRY}   ESPACE=pause | Q=quitter{RS}\n")
    while True:
        if etat['pause']:
            cv2.imshow(fenetre, dessiner_frame(frame_pause,detections_courantes,etat))
            key=cv2.waitKey(30)&0xFF
            if key==ord(' '): etat['pause']=False; print(f"{GRN}Reprise{RS}")
            elif key in (ord('q'),ord('Q'),27): break
            continue
        ret,frame=cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
        now=time.time()
        if now-derniere_envoi>0.05 and not q_in.full(): q_in.put(frame.copy()); derniere_envoi=now
        try:
            nd=q_out.get_nowait(); detections_courantes=nd
            for det in nd:
                if det['valide'] and det['texte'] not in etat['plaques_vues']:
                    etat['plaques_vues'].add(det['texte']); etat['nb_valides']+=1
                    etat['derniere_valide']=det
                    print(f"  {GRN}[{datetime.now().strftime('%H:%M:%S')}] {B}{det['texte']}{RS} ({det['score']:.0%})")
        except queue.Empty: pass
        fa=dessiner_frame(frame,detections_courantes,etat)
        hf,wf=fa.shape[:2]
        if max(hf,wf)>1280: s=1280/max(hf,wf); fa=cv2.resize(fa,(int(wf*s),int(hf*s)))
        cv2.imshow(fenetre,fa)
        key=cv2.waitKey(1)&0xFF
        if key==ord(' '): etat['pause']=True; frame_pause=frame.copy(); print(f"{YLW}Pause{RS}")
        elif key in (ord('q'),ord('Q'),27): break
    worker.arreter(); cap.release(); cv2.destroyAllWindows()
    print(f"\n{CYAN}{B}Bilan : {etat['nb_valides']} plaque(s){RS}")
    for p in sorted(etat['plaques_vues']): print(f"  {GRN}* {p}{RS}")


def main():
    print(f"\n{CYAN}{B}Lecteur Plaques Belges - Detection renforcee (faible lumiere){RS}\n")
    parser=argparse.ArgumentParser()
    g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--webcam',metavar='ID',nargs='?',const=0,type=int)
    g.add_argument('--video',metavar='FICHIER')
    args=parser.parse_args()
    if args.webcam is not None:
        cap=cv2.VideoCapture(args.webcam)
        if not cap.isOpened(): print(f"{RED}Webcam introuvable{RS}"); sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        nom_source=f"Webcam #{args.webcam}"
    else:
        if not os.path.isfile(args.video): print(f"{RED}Fichier introuvable{RS}"); sys.exit(1)
        cap=cv2.VideoCapture(args.video)
        fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{CYAN}Video: {args.video} | {total} frames | {fps:.1f} FPS{RS}\n")
        nom_source=os.path.basename(args.video)
    boucle_video(cap, nom_source, args)

if __name__=='__main__':
    main()