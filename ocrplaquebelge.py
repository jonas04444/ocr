#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   🎥 Lecteur de Plaques Belges — EasyOCR Edition            ║
║   Bien plus précis que Tesseract sur vidéo / flou / angle   ║
║                                                              ║
║   Webcam : python plaque_video_easyocr.py --webcam          ║
║   MP4    : python plaque_video_easyocr.py --video film.mp4  ║
║                                                              ║
║   Touches :  ESPACE → pause/reprise   Q → quitter           ║
╚══════════════════════════════════════════════════════════════╝

Installation :
    pip install easyocr opencv-python numpy
    (EasyOCR télécharge ses modèles automatiquement au 1er lancement ~100 Mo)
"""

import sys, re, os, argparse, time, threading, queue
from datetime import datetime

import cv2
import numpy as np

# ── EasyOCR ──────────────────────────────────────────────────
try:
    import easyocr
except ImportError:
    print("[ERREUR] EasyOCR non installé.")
    print("  → pip install easyocr")
    sys.exit(1)

# ── Couleurs terminal ─────────────────────────────────────────
RS="\033[0m"; B="\033[1m"; CYAN="\033[96m"; GRN="\033[92m"
YLW="\033[93m"; RED="\033[91m"; GRY="\033[90m"

# ── Patterns plaques belges ───────────────────────────────────
PATTERN_NOUVELLE = re.compile(r'^[0-9]-[A-Z]{3}-[0-9]{3}$')
PATTERN_ANCIENNE = re.compile(r'^[A-Z]{1,3}-[0-9]{1,4}-[A-Z]{0,3}$')

# ── Couleurs affichage vidéo (BGR) ────────────────────────────
CLR_VALIDE   = ( 50, 220,  50)
CLR_CANDIDAT = (  0, 165, 255)
CLR_OVERLAY  = ( 20,  20,  20)
CLR_PAUSE    = (  0, 200, 255)


# ══════════════════════════════════════════════════════════════
#  INITIALISATION EASYOCR (une seule fois au démarrage)
# ══════════════════════════════════════════════════════════════

def init_reader():
    """
    Crée le reader EasyOCR.
    - lang=['en'] suffit pour les plaques (chiffres + lettres latines)
    - gpu=False : fonctionne sur tous les PC sans carte graphique
    - Les modèles (~100 Mo) sont téléchargés automatiquement la 1ère fois
      dans ~/.EasyOCR/
    """
    print(f"{CYAN}⚙  Chargement du modèle EasyOCR...{RS}")
    print(f"{GRY}   (1ère fois : téléchargement ~100 Mo, patientez){RS}\n")
    reader = easyocr.Reader(
        ['en'],
        gpu=False,
        # Désactive les messages verbeux d'EasyOCR
        verbose=False,
    )
    print(f"{GRN}✅ Modèle chargé.{RS}\n")
    return reader


# ══════════════════════════════════════════════════════════════
#  DÉTECTION ZONE PLAQUE (inchangée — elle fonctionne bien)
# ══════════════════════════════════════════════════════════════

def detecter_zone_plaque(img_bgr):
    """
    Localise les zones candidates pour une plaque :
      A. Filtre couleur blanc (plaques belges récentes)
      B. Filtre couleur jaune (particuliers)
      C. Gradient morphologique (robuste la nuit / distance)
    """
    h_img, w_img = img_bgr.shape[:2]
    candidates = []
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    masque_blanc = cv2.inRange(hsv, np.array([0,   0, 170]), np.array([180, 50, 255]))
    masque_jaune = cv2.inRange(hsv, np.array([18,  60, 100]), np.array([35, 255, 255]))

    for nom, masque in [("blanc", masque_blanc), ("jaune", masque_jaune)]:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, k)
        masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN,  k)
        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            aire = cv2.contourArea(c)
            if aire < 600:
                continue
            _, (rw, rh), _ = cv2.minAreaRect(c)
            if rw < rh: rw, rh = rh, rw
            if rh < 5:  continue
            ratio = rw / rh
            score_ratio = 1.0 - abs(ratio - 4.6) / 4.6
            if score_ratio < 0.18: continue
            score_aire = min(aire / (w_img * h_img * 0.15), 1.0)
            score = score_ratio * 0.7 + score_aire * 0.3
            x, y, w, hh = cv2.boundingRect(c)
            candidates.append({'x':x,'y':y,'w':w,'h':hh,'score':score,'methode':nom})

    # Gradient
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5,5), 0)
    gx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    mx = grad.max()
    if mx > 0:
        grad = np.uint8(np.clip(grad / mx * 255, 0, 255))
        _, seuil = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        seuil = cv2.morphologyEx(seuil, cv2.MORPH_CLOSE, k2)
        contours2, _ = cv2.findContours(seuil, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for c in contours2:
            if cv2.contourArea(c) < 600: continue
            x, y, w, hh = cv2.boundingRect(c)
            if hh == 0: continue
            ratio = w / hh
            sr = 1.0 - abs(ratio - 4.6) / 4.6
            if sr < 0.22: continue
            candidates.append({'x':x,'y':y,'w':w,'h':hh,'score':sr*0.5,'methode':'gradient'})

    candidates.sort(key=lambda c: -c['score'])
    return candidates[:6]


# ══════════════════════════════════════════════════════════════
#  PRÉTRAITEMENT ROI — amélioré pour EasyOCR
# ══════════════════════════════════════════════════════════════

def pretraiter_pour_easyocr(img_bgr, x, y, w, h):
    """
    Extrait et prépare la ROI de la plaque pour EasyOCR.

    Différences clés vs Tesseract :
      - EasyOCR préfère l'image en couleur BGR (pas de binarisation forcée)
      - On upscale à une largeur fixe de 400px (optimal pour le modèle)
      - CLAHE appliqué sur le canal L (Lab) pour préserver les couleurs
      - On retourne DEUX variantes : couleur + niveaux de gris rehaussés
    """
    marge = max(6, int(h * 0.12))
    x1 = max(0, x - marge);  y1 = max(0, y - marge)
    x2 = min(img_bgr.shape[1], x + w + marge)
    y2 = min(img_bgr.shape[0], y + h + marge)
    roi = img_bgr[y1:y2, x1:x2]

    if roi.size == 0:
        return []

    # ── Redimensionner à largeur cible 400px ──────────────────
    rh, rw = roi.shape[:2]
    if rw == 0: return []
    cible_w = 400
    scale   = cible_w / rw
    roi_r   = cv2.resize(roi, (cible_w, max(1, int(rh * scale))),
                         interpolation=cv2.INTER_CUBIC)

    # ── Variante 1 : couleur + CLAHE sur luminosité ───────────
    lab   = cv2.cvtColor(roi_r, cv2.COLOR_BGR2Lab)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b_ch])
    v1_color = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)

    # ── Variante 2 : niveaux de gris débruités + sharpen ─────
    gris = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)
    gris = cv2.fastNlMeansDenoising(gris, h=9,
                                    templateWindowSize=7, searchWindowSize=15)
    # Netteté (unsharp mask)
    blur  = cv2.GaussianBlur(gris, (0, 0), 2)
    sharp = cv2.addWeighted(gris, 1.6, blur, -0.6, 0)
    v2_gray = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    return [v1_color, v2_gray]


# ══════════════════════════════════════════════════════════════
#  CORRECTIONS CARACTÈRES (identique à l'original)
# ══════════════════════════════════════════════════════════════

def _forcer_chiffre(c):
    M = {'O':'0','Q':'0','D':'0','I':'1','L':'1','l':'1',
         'Z':'2','S':'5','G':'6','B':'8','g':'9','A':'4'}
    return M.get(c.upper(), c) if not c.isdigit() else c

def _forcer_lettre(c):
    M = {'0':'O','1':'I','5':'S','6':'G','8':'B','4':'A'}
    return M.get(c, c) if not c.isalpha() else c

def corriger_caracteres(texte):
    texte = re.sub(r'[\s_–—.]', '-', texte).strip('-').upper()
    # Supprimer double tirets
    texte = re.sub(r'-{2,}', '-', texte)
    if len(texte) == 9 and texte[1] == '-' and texte[5] == '-':
        chars = list(texte)
        chars[0] = _forcer_chiffre(chars[0])
        for i in (2, 3, 4): chars[i] = _forcer_lettre(chars[i])
        for i in (6, 7, 8): chars[i] = _forcer_chiffre(chars[i])
        return ''.join(chars)
    return texte

def scorer_plaque(texte):
    if not texte: return 0.0
    if PATTERN_NOUVELLE.match(texte): return 1.0
    if PATTERN_ANCIENNE.match(texte): return 0.85
    score = 0.0
    if len(texte) == 9: score += 0.3
    elif len(texte) == 7: score += 0.2
    if len(texte) > 1 and texte[1] == '-': score += 0.2
    if len(texte) > 5 and texte[5] == '-': score += 0.2
    if re.search(r'\d', texte):    score += 0.1
    if re.search(r'[A-Z]', texte): score += 0.1
    return min(score, 0.99)

def est_valide(texte):
    return bool(PATTERN_NOUVELLE.match(texte) or PATTERN_ANCIENNE.match(texte))


# ══════════════════════════════════════════════════════════════
#  OCR EASYOCR
# ══════════════════════════════════════════════════════════════

def ocr_easyocr(reader, variantes_roi):
    """
    Lance EasyOCR sur chaque variante de ROI.

    Paramètres clés EasyOCR pour plaques :
      - allowlist     : restreint aux caractères possibles (comme Tesseract whitelist)
      - paragraph     : False → chaque mot est traité séparément
      - min_size      : ignore les textes trop petits
      - contrast_ths  : seuil de contraste plus permissif pour plaques
      - adjust_contrast : rehaussement auto si contraste insuffisant
      - text_threshold : confiance minimale pour accepter un résultat
    """
    resultats = {}
    ALLOWLIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'

    for roi_bgr in variantes_roi:
        try:
            detections = reader.readtext(
                roi_bgr,
                allowlist      = ALLOWLIST,
                paragraph      = False,
                min_size       = 10,
                contrast_ths   = 0.2,       # plus permissif que défaut (0.1)
                adjust_contrast= 0.6,       # rehausse si contraste < 0.6
                text_threshold = 0.55,      # confiance min (défaut 0.7, on baisse)
                low_text       = 0.35,      # détecte les caractères peu contrastés
                width_ths      = 0.8,       # fusionne les mots proches
            )

            # Fusionner tous les fragments sur la même ligne
            fragments = [d[1].strip().upper() for d in detections if d[2] > 0.3]
            if not fragments:
                continue

            # Reconstituer le texte complet de la plaque
            texte_brut = ''.join(re.sub(r'[^A-Z0-9\-]', '', f) for f in fragments)
            if not texte_brut:
                continue

            texte = corriger_caracteres(texte_brut)

            # Si le format n'a pas de tirets, tenter de les insérer
            if '-' not in texte and len(texte) == 7:
                # Format nouvelle plaque sans tirets : 1ABC234 → 1-ABC-234
                texte = f"{texte[0]}-{texte[1:4]}-{texte[4:7]}"

            score = scorer_plaque(texte)

            if texte not in resultats or score > resultats[texte]['score']:
                # Confiance EasyOCR moyenne sur les fragments
                conf_moy = sum(d[2] for d in detections) / len(detections) if detections else 0
                resultats[texte] = {
                    'texte':  texte,
                    'score':  score,
                    'conf':   conf_moy,
                    'valide': est_valide(texte),
                }
        except Exception:
            continue

    return sorted(resultats.values(), key=lambda r: -r['score'])


# ══════════════════════════════════════════════════════════════
#  ANALYSER UNE FRAME
# ══════════════════════════════════════════════════════════════

def analyser_frame(frame, reader):
    """Pipeline complet sur une frame : détection → prétraitement → EasyOCR."""
    candidates = detecter_zone_plaque(frame)
    detections = []

    for cand in candidates:
        variantes = pretraiter_pour_easyocr(
            frame, cand['x'], cand['y'], cand['w'], cand['h']
        )
        if not variantes:
            continue
        resultats = ocr_easyocr(reader, variantes)
        if resultats:
            best = resultats[0]
            detections.append({**best,
                                'x': cand['x'], 'y': cand['y'],
                                'w': cand['w'],  'h': cand['h']})
    return detections


# ══════════════════════════════════════════════════════════════
#  RENDU HUD
# ══════════════════════════════════════════════════════════════

def dessiner_frame(frame, detections, etat):
    out  = frame.copy()
    hf, wf = out.shape[:2]

    for det in detections:
        x, y, w, hh  = det['x'], det['y'], det['w'], det['h']
        couleur      = CLR_VALIDE if det['valide'] else CLR_CANDIDAT
        epaisseur    = 3 if det['valide'] else 2

        cv2.rectangle(out, (x, y), (x+w, y+hh), couleur, epaisseur)

        # Coins stylisés
        tc = min(w, hh) // 4
        for (px, py), (dx, dy) in [
            ((x,   y),      ( 1,  1)), ((x+w, y),      (-1,  1)),
            ((x,   y+hh),   ( 1, -1)), ((x+w, y+hh),   (-1, -1)),
        ]:
            cv2.line(out, (px, py), (px + dx*tc, py),    couleur, 4)
            cv2.line(out, (px, py), (px, py + dy*tc),    couleur, 4)

        label = f"{det['texte']}  {det['score']:.0%}"
        font  = cv2.FONT_HERSHEY_DUPLEX
        (lw, lh), bl = cv2.getTextSize(label, font, 0.9, 2)
        ly = max(y - 12, lh + 8)
        overlay = out.copy()
        cv2.rectangle(overlay, (x, ly-lh-8), (x+lw+12, ly+bl+2), couleur, -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        cv2.putText(out, label, (x+6, ly-2), font, 0.9, (10,10,10), 2)

    # ── HUD bas ───────────────────────────────────────────────
    hud_h = 54
    ov2 = out.copy()
    cv2.rectangle(ov2, (0, hf-hud_h), (wf, hf), CLR_OVERLAY, -1)
    cv2.addWeighted(ov2, 0.70, out, 0.30, 0, out)

    cv2.putText(out, f"SOURCE: {etat['source']}",
                (12, hf-hud_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170,170,170), 1)
    cv2.putText(out, f"MOTEUR: EasyOCR",
                (12, hf-hud_h+38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100,200,100), 1)
    cv2.putText(out, f"UNIQUES: {etat['nb_valides']}",
                (220, hf-hud_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170,170,170), 1)

    if etat['derniere_valide']:
        txt = f"DERNIERE: {etat['derniere_valide']['texte']}"
        (tw,_),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.78, 2)
        cv2.putText(out, txt, (wf//2 - tw//2, hf-8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.78, CLR_VALIDE, 2)

    if etat['pause']:
        ptxt = "|| PAUSE  [ESPACE pour reprendre]"
        (pw,_),_ = cv2.getTextSize(ptxt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(out, ptxt, (wf-pw-12, hf-hud_h+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_PAUSE, 2)
    else:
        rtxt = "ESPACE=pause  |  Q=quitter"
        (rw,_),_ = cv2.getTextSize(rtxt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(out, rtxt, (wf-rw-12, hf-hud_h+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120,120,120), 1)

    cv2.putText(out, datetime.now().strftime("%H:%M:%S"),
                (wf-88, hf-8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140,140,140), 1)
    return out


# ══════════════════════════════════════════════════════════════
#  WORKER OCR (thread séparé)
# ══════════════════════════════════════════════════════════════

class WorkerOCR(threading.Thread):
    def __init__(self, q_in, q_out, reader):
        super().__init__(daemon=True)
        self.q_in  = q_in
        self.q_out = q_out
        self.reader = reader
        self.actif  = True

    def run(self):
        while self.actif:
            try:
                frame = self.q_in.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                dets = analyser_frame(frame, self.reader)
                while not self.q_out.empty():
                    try: self.q_out.get_nowait()
                    except queue.Empty: break
                self.q_out.put(dets)
            except Exception:
                pass

    def arreter(self):
        self.actif = False


# ══════════════════════════════════════════════════════════════
#  BOUCLE VIDÉO PRINCIPALE
# ══════════════════════════════════════════════════════════════

def boucle_video(cap, nom_source, reader):
    q_in   = queue.Queue(maxsize=2)
    q_out  = queue.Queue(maxsize=2)
    worker = WorkerOCR(q_in, q_out, reader)
    worker.start()

    etat = {
        'source':          nom_source,
        'pause':           False,
        'nb_valides':      0,
        'derniere_valide': None,
        'plaques_vues':    set(),
    }

    detections_courantes = []
    frame_pause          = None
    derniere_envoi       = 0
    fenetre              = "🚗 Plaques Belges — EasyOCR"

    print(f"\n{CYAN}{B}🎥 Lecture démarrée : {nom_source}{RS}")
    print(f"{GRY}   ESPACE = pause/reprise  |  Q = quitter{RS}\n")

    while True:
        # ── Pause ────────────────────────────────────────────
        if etat['pause']:
            cv2.imshow(fenetre, dessiner_frame(frame_pause, detections_courantes, etat))
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                etat['pause'] = False
                print(f"{GRN}▶  Reprise{RS}")
            elif key in (ord('q'), ord('Q'), 27):
                break
            continue

        # ── Lecture frame ─────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # ── Envoi au worker ───────────────────────────────────
        now = time.time()
        if now - derniere_envoi > 0.08 and not q_in.full():
            q_in.put(frame.copy())
            derniere_envoi = now

        # ── Récupérer résultats ───────────────────────────────
        try:
            nouvelles = q_out.get_nowait()
            detections_courantes = nouvelles
            for det in nouvelles:
                if det['valide'] and det['texte'] not in etat['plaques_vues']:
                    etat['plaques_vues'].add(det['texte'])
                    etat['nb_valides']     += 1
                    etat['derniere_valide'] = det
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"  {GRN}✅ [{ts}]  {B}{det['texte']}{RS}"
                          f"  (score {det['score']:.0%},"
                          f" conf EasyOCR {det['conf']:.0%})")
        except queue.Empty:
            pass

        # ── Affichage ─────────────────────────────────────────
        frame_out = dessiner_frame(frame, detections_courantes, etat)
        hf, wf = frame_out.shape[:2]
        if max(hf, wf) > 1280:
            s = 1280 / max(hf, wf)
            frame_out = cv2.resize(frame_out, (int(wf*s), int(hf*s)))
        cv2.imshow(fenetre, frame_out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            etat['pause']  = True
            frame_pause    = frame.copy()
            print(f"{YLW}⏸  Pause{RS}")
        elif key in (ord('q'), ord('Q'), 27):
            break

    worker.arreter()
    cap.release()
    cv2.destroyAllWindows()

    # ── Bilan ─────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"{CYAN}{B}📊 Bilan de session — EasyOCR{RS}")
    print(f"   Plaques uniques : {etat['nb_valides']}")
    for p in sorted(etat['plaques_vues']):
        print(f"   {GRN}• {p}{RS}")
    print()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def banner():
    print(f"""
{CYAN}{B}╔══════════════════════════════════════════════════════╗
║   🎥 Lecteur Plaques Belges — EasyOCR Edition       ║
║   Précision améliorée pour vidéo / flou / distance  ║
╚══════════════════════════════════════════════════════╝{RS}
""")

def main():
    banner()

    parser = argparse.ArgumentParser(
        description="Détection plaques belges — moteur EasyOCR",
        formatter_class=argparse.RawTextHelpFormatter
    )
    groupe = parser.add_mutually_exclusive_group(required=True)
    groupe.add_argument('--webcam', metavar='ID', nargs='?', const=0, type=int,
                        help='Webcam (0 par défaut)')
    groupe.add_argument('--video', metavar='FICHIER',
                        help='Fichier vidéo (mp4, avi, mkv...)')
    args = parser.parse_args()

    # ── Initialisation EasyOCR ────────────────────────────────
    reader = init_reader()

    # ── Ouverture source ──────────────────────────────────────
    if args.webcam is not None:
        idx = args.webcam
        print(f"{CYAN}📷 Ouverture webcam #{idx}...{RS}")
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"{RED}[ERREUR] Webcam #{idx} introuvable.{RS}")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        nom_source = f"Webcam #{idx}"
    else:
        chemin = args.video
        if not os.path.isfile(chemin):
            print(f"{RED}[ERREUR] Fichier introuvable : {chemin}{RS}")
            sys.exit(1)
        cap = cv2.VideoCapture(chemin)
        if not cap.isOpened():
            print(f"{RED}[ERREUR] Impossible d'ouvrir : {chemin}{RS}")
            sys.exit(1)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{CYAN}🎬 Vidéo : {chemin}{RS}")
        print(f"   {total} frames  |  {fps:.1f} FPS  |  "
              f"{total/fps:.1f}s\n" if fps > 0 else "\n")
        nom_source = os.path.basename(chemin)

    boucle_video(cap, nom_source, reader)


if __name__ == '__main__':
    main()