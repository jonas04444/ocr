#!/usr/bin/env python3
"""
Lecteur de Plaques Belges - Video/Webcam
Usage:
    python ocrvideoplaque.py --video fichier.mp4
    python ocrvideoplaque.py --webcam
Touches: ESPACE = pause/reprise  |  Q = quitter
"""

import sys, re, os, argparse, time, threading, queue
from datetime import datetime
import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    print("[ERREUR] pip install pytesseract opencv-python numpy")
    sys.exit(1)

# ── Couleurs terminal ─────────────────────────────────────────
RS="\033[0m"; B="\033[1m"; CYAN="\033[96m"; GRN="\033[92m"
YLW="\033[93m"; RED="\033[91m"; GRY="\033[90m"

# ── Patterns plaques belges ───────────────────────────────────
PATTERN_NOUVELLE = re.compile(r'^[0-9]-[A-Z]{3}-[0-9]{3}$')
PATTERN_ANCIENNE = re.compile(r'^[A-Z]{1,3}-[0-9]{1,4}-[A-Z]{0,3}$')

# ── Config Tesseract ──────────────────────────────────────────
TESS_CFG = [
    r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
    r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
    r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
]

CLR_VALIDE   = (50, 220, 50)
CLR_CANDIDAT = (0, 165, 255)
CLR_OVERLAY  = (20, 20, 20)
CLR_PAUSE    = (0, 200, 255)


# ══════════════════════════════════════════════════════════════
#  DÉTECTION
# ══════════════════════════════════════════════════════════════

def ameliorer_luminosite(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(v)
    return cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)


def detecter_plaques(img_bgr):
    """Retourne liste de {x,y,w,h,score}"""
    H, W = img_bgr.shape[:2]
    candidates = []
    img_eq = ameliorer_luminosite(img_bgr)
    hsv = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)

    # Masques couleur
    for lo, hi in [
        ([0,0,100],   [180,60,255]),  # blanc large
        ([15,40,80],  [40,255,255]),  # jaune
    ]:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT,(20,6)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT,(10,4)))
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            aire = cv2.contourArea(c)
            if aire < 500: continue
            _, (rw,rh), _ = cv2.minAreaRect(c)
            if rw < rh: rw,rh = rh,rw
            if rh < 4: continue
            ratio = rw/rh
            sr = 1.0 - abs(ratio - 4.6)/4.6
            if sr < 0.20: continue
            x,y,w,hh = cv2.boundingRect(c)
            # Rejeter si trop grand
            if w > W*0.85 or hh > H*0.60: continue
            candidates.append({'x':x,'y':y,'w':w,'h':hh,'score':sr})

    # Gradient
    gris = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5,5), 0)
    gx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    mx = grad.max()
    if mx > 0:
        grad = np.uint8(np.clip(grad/mx*255, 0, 255))
        _, seuil = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        seuil = cv2.morphologyEx(seuil, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT,(25,3)))
        for c in cv2.findContours(seuil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) < 500: continue
            x,y,w,hh = cv2.boundingRect(c)
            if hh == 0: continue
            ratio = w/hh
            sr = 1.0 - abs(ratio - 4.6)/4.6
            if sr < 0.22: continue
            if w > W*0.85 or hh > H*0.60: continue
            candidates.append({'x':x,'y':y,'w':w,'h':hh,'score':sr*0.5})

    # Dédoublonnage IoU
    gardes = []
    for c in sorted(candidates, key=lambda x: -x['score']):
        if not any(_iou(c,g) > 0.4 for g in gardes):
            gardes.append(c)

    gardes.sort(key=lambda c: -c['score'])
    return gardes[:5]


def _iou(a, b):
    x1=max(a['x'],b['x']); y1=max(a['y'],b['y'])
    x2=min(a['x']+a['w'],b['x']+b['w']); y2=min(a['y']+a['h'],b['y']+b['h'])
    inter=max(0,x2-x1)*max(0,y2-y1)
    if inter==0: return 0.0
    return inter/(a['w']*a['h']+b['w']*b['h']-inter)


# ══════════════════════════════════════════════════════════════
#  PRÉTRAITEMENT + OCR
# ══════════════════════════════════════════════════════════════

def pretraiter(img_bgr, x, y, w, h):
    """Extrait et prépare la ROI pour Tesseract."""
    marge = max(8, int(h*0.1))
    x1=max(0,x-marge); y1=max(0,y-marge)
    x2=min(img_bgr.shape[1],x+w+marge); y2=min(img_bgr.shape[0],y+h+marge)
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0: return {}

    # Upscale x3
    rh,rw = roi.shape[:2]
    roi = cv2.resize(roi, (rw*3, rh*3), interpolation=cv2.INTER_CUBIC)

    variantes = {}

    # V1 : canal vert (rouge → noir, blanc → blanc = contraste max)
    cg = roi[:,:,1].copy()
    cg = cv2.fastNlMeansDenoising(cg, h=8)
    _, v1 = cv2.threshold(cg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    variantes['vert'] = v1
    variantes['vert_inv'] = cv2.bitwise_not(v1)

    # V2 : gris + CLAHE
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gris = cv2.fastNlMeansDenoising(gris, h=10)
    gc = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4,4)).apply(gris)
    _, v2 = cv2.threshold(gc, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    variantes['otsu'] = v2
    variantes['otsu_inv'] = cv2.bitwise_not(v2)

    # V3 : adaptatif
    variantes['adapt'] = cv2.adaptiveThreshold(
        gc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 9)

    return variantes


def _forcer_chiffre(c):
    M={'O':'0','Q':'0','D':'0','I':'1','L':'1','Z':'2','S':'5','G':'6','B':'8','A':'4'}
    return M.get(c,c) if not c.isdigit() else c

def _forcer_lettre(c):
    M={'0':'O','1':'I','5':'S','6':'G','8':'B','4':'A'}
    return M.get(c,c) if not c.isalpha() else c

def corriger(texte):
    texte = re.sub(r'[\s_]','-',texte).strip('-')
    texte = re.sub(r'-{2,}','-',texte)
    # Reconstruire si 7 chars sans tirets
    if len(texte)==7 and '-' not in texte:
        texte = f"{texte[0]}-{texte[1:4]}-{texte[4:7]}"
    if len(texte)==9 and texte[1]=='-' and texte[5]=='-':
        c = list(texte)
        c[0] = _forcer_chiffre(c[0])
        for i in (2,3,4): c[i] = _forcer_lettre(c[i])
        for i in (6,7,8): c[i] = _forcer_chiffre(c[i])
        if c[8]=='-': c[8]='4'
        return ''.join(c)
    return texte

def scorer(texte):
    if not texte: return 0.0
    if PATTERN_NOUVELLE.match(texte): return 1.0
    if PATTERN_ANCIENNE.match(texte): return 0.85
    s=0.0
    if len(texte)==9: s+=0.3
    elif len(texte)==7: s+=0.2
    if len(texte)>1 and texte[1]=='-': s+=0.2
    if len(texte)>5 and texte[5]=='-': s+=0.2
    if re.search(r'\d',texte): s+=0.1
    if re.search(r'[A-Z]',texte): s+=0.1
    return min(s,0.99)

def est_valide(texte):
    return bool(PATTERN_NOUVELLE.match(texte) or PATTERN_ANCIENNE.match(texte))

def lire_plaque(variantes):
    """Lance Tesseract sur toutes les variantes, retourne le meilleur résultat."""
    meilleurs = {}
    for nom, img in variantes.items():
        img_m = cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,value=255)
        for cfg in TESS_CFG:
            try:
                brut = pytesseract.image_to_string(img_m, config=cfg, lang='eng')
                texte = re.sub(r'[^A-Z0-9\-]','', brut.strip().upper())
                if not texte: continue
                texte = corriger(texte)
                sc = scorer(texte)
                if texte not in meilleurs or sc > meilleurs[texte]['score']:
                    meilleurs[texte] = {'texte':texte,'score':sc,'valide':est_valide(texte)}
            except Exception as e:
                print(f"  [OCR err] {e}")
                continue
    if not meilleurs: return None
    return sorted(meilleurs.values(), key=lambda r:-r['score'])[0]


def analyser_frame(frame):
    """Analyse complète d'une frame. Retourne liste de détections."""
    zones = detecter_plaques(frame)
    resultats = []
    for z in zones:
        variantes = pretraiter(frame, z['x'], z['y'], z['w'], z['h'])
        if not variantes:
            resultats.append({'texte':'','score':0,'valide':False,'ocr_ok':False,
                               'x':z['x'],'y':z['y'],'w':z['w'],'h':z['h']})
            continue
        res = lire_plaque(variantes)
        if res and res['score'] >= 0.4:
            resultats.append({**res,'ocr_ok':True,
                               'x':z['x'],'y':z['y'],'w':z['w'],'h':z['h']})
        else:
            resultats.append({'texte':'','score':0,'valide':False,'ocr_ok':False,
                               'x':z['x'],'y':z['y'],'w':z['w'],'h':z['h']})
    return resultats


# ══════════════════════════════════════════════════════════════
#  AFFICHAGE
# ══════════════════════════════════════════════════════════════

def dessiner(frame, detections, etat):
    out = frame.copy()
    H, W = out.shape[:2]

    for det in detections:
        x,y,w,h = det['x'],det['y'],det['w'],det['h']
        col = CLR_VALIDE if det['valide'] else CLR_CANDIDAT
        ep  = 3 if det['valide'] else 2
        cv2.rectangle(out,(x,y),(x+w,y+h),col,ep)

        # Coins stylisés
        tc = min(w,h)//5
        for (px,py),(dx,dy) in [((x,y),(1,1)),((x+w,y),(-1,1)),
                                  ((x,y+h),(1,-1)),((x+w,y+h),(-1,-1))]:
            cv2.line(out,(px,py),(px+dx*tc,py),col,3)
            cv2.line(out,(px,py),(px,py+dy*tc),col,3)

        # Label
        if det['ocr_ok'] and det['texte']:
            label = f"{det['texte']}  {det['score']:.0%}"
            (lw,lh),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
            # Position au-dessus ou à l'intérieur
            lx = max(0, min(x, W-lw-14))
            ly = y-10 if y > lh+14 else y+lh+12
            ov = out.copy()
            cv2.rectangle(ov,(lx,ly-lh-6),(lx+lw+12,ly+4),col,-1)
            cv2.addWeighted(ov,0.8,out,0.2,0,out)
            cv2.putText(out,label,(lx+6,ly),cv2.FONT_HERSHEY_DUPLEX,0.85,(10,10,10),2)

    # HUD
    hud = 50
    ov2 = out.copy()
    cv2.rectangle(ov2,(0,H-hud),(W,H),CLR_OVERLAY,-1)
    cv2.addWeighted(ov2,0.7,out,0.3,0,out)
    cv2.putText(out,f"SOURCE: {etat['source']}",(10,H-hud+18),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(170,170,170),1)
    cv2.putText(out,f"PLAQUES: {etat['nb_valides']}",(10,H-hud+36),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(170,170,170),1)
    if etat['derniere_valide']:
        txt = f">> {etat['derniere_valide']['texte']} <<"
        (tw,_),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_DUPLEX,0.85,2)
        cv2.putText(out,txt,(W//2-tw//2,H-8),cv2.FONT_HERSHEY_DUPLEX,0.85,CLR_VALIDE,2)
    statut = "|| PAUSE [ESPACE]" if etat['pause'] else "ESPACE=pause | Q=quitter"
    col_s  = CLR_PAUSE if etat['pause'] else (120,120,120)
    (sw,_),_ = cv2.getTextSize(statut,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
    cv2.putText(out,statut,(W-sw-10,H-hud+18),cv2.FONT_HERSHEY_SIMPLEX,0.5,col_s,1)
    cv2.putText(out,datetime.now().strftime("%H:%M:%S"),(W-88,H-8),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(140,140,140),1)
    return out


# ══════════════════════════════════════════════════════════════
#  WORKER THREAD
# ══════════════════════════════════════════════════════════════

class WorkerOCR(threading.Thread):
    def __init__(self, q_in, q_out):
        super().__init__(daemon=True)
        self.q_in = q_in; self.q_out = q_out; self.actif = True

    def run(self):
        while self.actif:
            try:
                frame = self.q_in.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                dets = analyser_frame(frame)
                while not self.q_out.empty():
                    try: self.q_out.get_nowait()
                    except queue.Empty: break
                self.q_out.put(dets)
            except Exception as e:
                print(f"{RED}[Worker] ERREUR : {e}{RS}")
                import traceback; traceback.print_exc()

    def arreter(self): self.actif = False


# ══════════════════════════════════════════════════════════════
#  BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════

def boucle(cap, nom_source):
    q_in  = queue.Queue(maxsize=2)
    q_out = queue.Queue(maxsize=2)
    worker = WorkerOCR(q_in, q_out)
    worker.start()

    etat = {'source':nom_source,'pause':False,'nb_valides':0,
            'derniere_valide':None,'plaques_vues':set()}

    # Détections visuelles (zones orange immédiates)
    zones_visuelles = []
    # Dernières détections OCR reçues
    dets_ocr = []

    frame_pause   = None
    dernier_envoi = 0
    fenetre       = "Plaques Belges"

    print(f"\n{CYAN}Lecture : {nom_source}{RS}")
    print(f"{GRY}ESPACE=pause | Q=quitter{RS}\n")

    while True:
        # ── Pause ──────────────────────────────────────────────
        if etat['pause']:
            cv2.imshow(fenetre, dessiner(frame_pause, dets_ocr, etat))
            k = cv2.waitKey(30) & 0xFF
            if k == ord(' '):
                etat['pause'] = False
                print(f"{GRN}Reprise{RS}")
            elif k in (ord('q'),ord('Q'),27): break
            continue

        # ── Lecture frame ───────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # ── Détection visuelle directe (rectangles immédiats) ──
        zones_visuelles = detecter_plaques(frame)
        affichage = [{'texte':'','score':0,'valide':False,'ocr_ok':False,
                      'x':z['x'],'y':z['y'],'w':z['w'],'h':z['h']}
                     for z in zones_visuelles]

        # ── Envoi au worker OCR ─────────────────────────────────
        now = time.time()
        if now - dernier_envoi > 0.1 and not q_in.full():
            q_in.put(frame.copy())
            dernier_envoi = now

        # ── Récupérer résultats OCR ─────────────────────────────
        try:
            dets_ocr = q_out.get_nowait()
            for det in dets_ocr:
                if det['ocr_ok'] and det['texte']:
                    print(f"  OCR: [{det['texte']}] score={det['score']:.0%} valide={det['valide']}")
                if det['valide'] and det['texte'] not in etat['plaques_vues']:
                    etat['plaques_vues'].add(det['texte'])
                    etat['nb_valides'] += 1
                    etat['derniere_valide'] = det
                    ts = datetime.now().strftime('%H:%M:%S')
                    print(f"  {GRN}✅ [{ts}] {B}{det['texte']}{RS} ({det['score']:.0%})")
        except queue.Empty:
            dets_ocr = affichage   # afficher zones visuelles en attendant

        # ── Rendu ───────────────────────────────────────────────
        fa = dessiner(frame, dets_ocr, etat)
        H, W = fa.shape[:2]
        if max(H,W) > 1280:
            s = 1280/max(H,W)
            fa = cv2.resize(fa,(int(W*s),int(H*s)))
        cv2.imshow(fenetre, fa)

        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            etat['pause'] = True
            frame_pause   = frame.copy()
            print(f"{YLW}Pause{RS}")
        elif k in (ord('q'),ord('Q'),27): break

    worker.arreter()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{CYAN}Bilan : {etat['nb_valides']} plaque(s){RS}")
    for p in sorted(etat['plaques_vues']):
        print(f"  {GRN}* {p}{RS}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print(f"\n{CYAN}{B}Lecteur Plaques Belges{RS}\n")
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--webcam', metavar='ID', nargs='?', const=0, type=int)
    g.add_argument('--video',  metavar='FICHIER')
    args = parser.parse_args()

    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            print(f"{RED}Webcam introuvable{RS}"); sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        nom = f"Webcam #{args.webcam}"
    else:
        if not os.path.isfile(args.video):
            print(f"{RED}Fichier introuvable : {args.video}{RS}"); sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"{RED}Impossible d'ouvrir la vidéo{RS}"); sys.exit(1)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {args.video} | {total} frames | {fps:.1f} FPS\n")
        nom = os.path.basename(args.video)

    boucle(cap, nom)

if __name__ == '__main__':
    main()