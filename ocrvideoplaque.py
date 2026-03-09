#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   🎥 Lecteur de Plaques Belges — Vidéo / Webcam             ║
║   Webcam : python plaque_video.py --webcam                  ║
║   MP4    : python plaque_video.py --video fichier.mp4       ║
║                                                              ║
║   Touches :  ESPACE → pause/reprise   Q → quitter           ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, re, os, argparse, time, threading, queue
from collections import deque
from datetime import datetime

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    print("[ERREUR] pip install pytesseract pillow opencv-python")
    sys.exit(1)

# ── Couleurs terminal ─────────────────────────────────────────
RS = "\033[0m"; B = "\033[1m"; CYAN = "\033[96m"; GRN = "\033[92m"
YLW = "\033[93m"; RED = "\033[91m"; GRY = "\033[90m"

# ── Patterns plaques belges ───────────────────────────────────
PATTERN_NOUVELLE = re.compile(r'^[0-9]-[A-Z]{3}-[0-9]{3}$')
PATTERN_ANCIENNE = re.compile(r'^[A-Z]{1,3}-[0-9]{1,4}-[A-Z]{0,3}$')

# ── Config Tesseract (whitelist stricte) ──────────────────────
TESS_CONFIGS = [
    r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
    r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
]

# ── Couleurs affichage vidéo (BGR) ────────────────────────────
CLR_VALIDE   = (50,  220,  50)   # vert
CLR_CANDIDAT = (0,   165, 255)   # orange
CLR_OVERLAY  = (20,   20,  20)   # fond HUD
CLR_PAUSE    = (0,   200, 255)   # jaune vif


# ══════════════════════════════════════════════════════════════
#  MOTEUR DE DÉTECTION (identique à plaque_belge.py)
# ══════════════════════════════════════════════════════════════

def detecter_zone_plaque(img_bgr):
    h_img, w_img = img_bgr.shape[:2]
    candidates = []
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    masque_blanc = cv2.inRange(hsv, np.array([0,   0, 180]), np.array([180, 40, 255]))
    masque_jaune = cv2.inRange(hsv, np.array([18, 60, 100]), np.array([35, 255, 255]))

    for nom, masque in [("blanc", masque_blanc), ("jaune", masque_jaune)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, kernel)
        masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN,  kernel)
        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            aire = cv2.contourArea(c)
            if aire < 800:
                continue
            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw
            if rh < 5:
                continue
            ratio = rw / rh
            score_ratio = 1.0 - abs(ratio - 4.6) / 4.6
            if score_ratio < 0.2:
                continue
            score_aire = min(aire / (w_img * h_img * 0.15), 1.0)
            score = score_ratio * 0.7 + score_aire * 0.3
            x, y, w, hh = cv2.boundingRect(c)
            candidates.append({'x': x, 'y': y, 'w': w, 'h': hh,
                                'score': score, 'methode': nom})

    # Méthode gradient
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    grad_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    grad   = cv2.magnitude(grad_x, grad_y)
    max_g  = grad.max()
    if max_g > 0:
        grad = np.uint8(np.clip(grad / max_g * 255, 0, 255))
        _, seuil = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        seuil = cv2.morphologyEx(seuil, cv2.MORPH_CLOSE, k2)
        contours2, _ = cv2.findContours(seuil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours2:
            aire = cv2.contourArea(c)
            if aire < 800:
                continue
            x, y, w, hh = cv2.boundingRect(c)
            if hh == 0:
                continue
            ratio = w / hh
            score_ratio = 1.0 - abs(ratio - 4.6) / 4.6
            if score_ratio < 0.25:
                continue
            candidates.append({'x': x, 'y': y, 'w': w, 'h': hh,
                                'score': score_ratio * 0.5, 'methode': 'gradient'})

    candidates.sort(key=lambda c: -c['score'])
    return candidates[:5]


def _ordonner_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def corriger_perspective(img_bgr, x, y, w, h, marge=8):
    x1 = max(0, x - marge);  y1 = max(0, y - marge)
    x2 = min(img_bgr.shape[1], x + w + marge)
    y2 = min(img_bgr.shape[0], y + h + marge)
    roi = img_bgr[y1:y2, x1:x2]
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return roi
    c_max   = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(c_max, True)
    poly    = cv2.approxPolyDP(c_max, epsilon, True)
    if len(poly) == 4:
        pts     = poly.reshape(4, 2).astype(np.float32)
        pts_ord = _ordonner_points(pts)
        tl, tr, br, bl = pts_ord
        larg = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        haut = int(max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br)))
        if larg > 0 and haut > 0:
            dst = np.array([[0,0],[larg-1,0],[larg-1,haut-1],[0,haut-1]], dtype=np.float32)
            M   = cv2.getPerspectiveTransform(pts_ord, dst)
            roi = cv2.warpPerspective(roi, M, (larg, haut))
    return roi


def pretraiter_plaque(roi_bgr):
    h, w = roi_bgr.shape[:2]
    if h == 0 or w == 0:
        return {}
    roi_bgr = cv2.resize(roi_bgr, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    gris    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gris    = cv2.fastNlMeansDenoising(gris, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gc      = clahe.apply(gris)
    _, v1   = cv2.threshold(gc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v3      = cv2.adaptiveThreshold(gc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 19, 9)
    return {'otsu': v1, 'otsu_inv': cv2.bitwise_not(v1), 'adaptatif': v3}


def _forcer_chiffre(c):
    M = {'O':'0','Q':'0','D':'0','I':'1','L':'1','l':'1',
         'Z':'2','S':'5','G':'6','B':'8','g':'9','A':'4'}
    return M.get(c, c) if not c.isdigit() else c

def _forcer_lettre(c):
    M = {'0':'O','1':'I','5':'S','6':'G','8':'B','4':'A'}
    return M.get(c, c) if not c.isalpha() else c

def corriger_caracteres(texte):
    texte = re.sub(r'[\s_–—]', '-', texte).strip('-')
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
    if re.search(r'\d', texte):   score += 0.1
    if re.search(r'[A-Z]', texte): score += 0.1
    return min(score, 0.99)

def est_valide(texte):
    return bool(PATTERN_NOUVELLE.match(texte) or PATTERN_ANCIENNE.match(texte))

def ocr_plaque(variantes):
    resultats = {}
    for _, img in variantes.items():
        img_m = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        for cfg in TESS_CONFIGS:
            try:
                texte = pytesseract.image_to_string(img_m, config=cfg, lang='eng').strip().upper()
                texte = re.sub(r'[^A-Z0-9\-]', '', texte)
                if not texte: continue
                texte = corriger_caracteres(texte)
                score = scorer_plaque(texte)
                if texte not in resultats or score > resultats[texte]['score']:
                    resultats[texte] = {'texte': texte, 'score': score, 'valide': est_valide(texte)}
            except Exception:
                continue
    return sorted(resultats.values(), key=lambda r: -r['score'])

def analyser_frame(frame):
    """Détecte et lit toutes les plaques dans une frame. Retourne les résultats."""
    candidates  = detecter_zone_plaque(frame)
    detections  = []   # [{texte, score, valide, x, y, w, h}]

    for cand in candidates:
        roi      = corriger_perspective(frame, cand['x'], cand['y'], cand['w'], cand['h'])
        variantes = pretraiter_plaque(roi)
        resultats = ocr_plaque(variantes)
        if resultats:
            best = resultats[0]
            detections.append({**best,
                                'x': cand['x'], 'y': cand['y'],
                                'w': cand['w'],  'h': cand['h']})
    return detections


# ══════════════════════════════════════════════════════════════
#  RENDU HUD SUR LA FRAME
# ══════════════════════════════════════════════════════════════

def dessiner_frame(frame, detections, etat):
    """
    Superpose sur la frame :
      - Rectangles autour des plaques (vert = valide, orange = candidat)
      - Labels avec le texte et le score
      - HUD en bas (source, statut pause, dernière plaque valide)
    """
    out = frame.copy()
    h_f, w_f = out.shape[:2]

    # ── Rectangles + labels ────────────────────────────────────
    for det in detections:
        x, y, w, hh = det['x'], det['y'], det['w'], det['h']
        couleur = CLR_VALIDE if det['valide'] else CLR_CANDIDAT
        epaisseur = 3 if det['valide'] else 2

        # Rectangle plaque
        cv2.rectangle(out, (x, y), (x + w, y + hh), couleur, epaisseur)

        # Coins épaissis (style moderne)
        taille_coin = min(w, hh) // 4
        for (px, py), (dx, dy) in [
            ((x, y),         (1, 1)),
            ((x + w, y),     (-1, 1)),
            ((x, y + hh),    (1, -1)),
            ((x + w, y + hh),(-1, -1)),
        ]:
            cv2.line(out, (px, py), (px + dx * taille_coin, py), couleur, 4)
            cv2.line(out, (px, py), (px, py + dy * taille_coin), couleur, 4)

        # Label avec fond semi-transparent
        label = f"{det['texte']}  {det['score']:.0%}"
        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.9
        thick = 2
        (lw, lh), bl = cv2.getTextSize(label, font, scale, thick)
        ly = max(y - 12, lh + 8)

        # Fond du label
        overlay = out.copy()
        cv2.rectangle(overlay, (x, ly - lh - 8), (x + lw + 12, ly + bl + 2), couleur, -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        cv2.putText(out, label, (x + 6, ly - 2), font, scale, (10, 10, 10), thick)

    # ── HUD bas de l'écran ─────────────────────────────────────
    hud_h = 52
    overlay2 = out.copy()
    cv2.rectangle(overlay2, (0, h_f - hud_h), (w_f, h_f), CLR_OVERLAY, -1)
    cv2.addWeighted(overlay2, 0.7, out, 0.3, 0, out)

    # Source (webcam / vidéo)
    src_txt = f"SOURCE: {etat['source']}"
    cv2.putText(out, src_txt, (12, h_f - hud_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Nb plaques valides détectées depuis le début
    nb_txt = f"PLAQUES: {etat['nb_valides']}"
    cv2.putText(out, nb_txt, (12, h_f - hud_h + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Dernière plaque valide (centre)
    if etat['derniere_valide']:
        dv = etat['derniere_valide']
        txt_dv = f"DERNIERE: {dv['texte']}"
        (tw, _), _ = cv2.getTextSize(txt_dv, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        cv2.putText(out, txt_dv, (w_f // 2 - tw // 2, h_f - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, CLR_VALIDE, 2)

    # Statut pause (droite)
    if etat['pause']:
        ptxt = "⏸  PAUSE  [ESPACE pour reprendre]"
        (pw, _), _ = cv2.getTextSize(ptxt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(out, ptxt, (w_f - pw - 12, h_f - hud_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_PAUSE, 2)
    else:
        rtxt = "▶  EN DIRECT  [ESPACE = pause  |  Q = quitter]"
        (rw, _), _ = cv2.getTextSize(rtxt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(out, rtxt, (w_f - rw - 12, h_f - hud_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)

    # Heure courante (droite bas)
    heure = datetime.now().strftime("%H:%M:%S")
    cv2.putText(out, heure, (w_f - 90, h_f - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return out


# ══════════════════════════════════════════════════════════════
#  WORKER OCR (thread séparé pour ne pas bloquer l'affichage)
# ══════════════════════════════════════════════════════════════

class WorkerOCR(threading.Thread):
    """
    Thread dédié à l'OCR.
    Reçoit des frames via une queue, publie les résultats dans une autre.
    """
    def __init__(self, q_in, q_out):
        super().__init__(daemon=True)
        self.q_in  = q_in
        self.q_out = q_out
        self.actif = True

    def run(self):
        while self.actif:
            try:
                frame = self.q_in.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                dets = analyser_frame(frame)
                # Vider la queue de sortie et mettre le résultat frais
                while not self.q_out.empty():
                    try: self.q_out.get_nowait()
                    except queue.Empty: break
                self.q_out.put(dets)
            except Exception as e:
                pass

    def arreter(self):
        self.actif = False


# ══════════════════════════════════════════════════════════════
#  BOUCLE PRINCIPALE VIDÉO
# ══════════════════════════════════════════════════════════════

def boucle_video(cap, nom_source, args):
    """
    Boucle principale :
      - Lit les frames de `cap` (webcam ou fichier)
      - Envoie chaque frame au worker OCR
      - Affiche le résultat le plus récent en overlay
      - Gère ESPACE (pause) et Q (quitter)
    """
    # Queues pour le worker OCR
    q_in  = queue.Queue(maxsize=2)
    q_out = queue.Queue(maxsize=2)
    worker = WorkerOCR(q_in, q_out)
    worker.start()

    # État global
    etat = {
        'source':          nom_source,
        'pause':           False,
        'nb_valides':      0,
        'derniere_valide': None,
        'plaques_vues':    set(),   # anti-doublon terminal
    }

    detections_courantes = []
    frame_pause          = None    # frame gelée pendant la pause
    derniere_envoi       = 0       # timestamp dernier envoi au worker

    print(f"\n{CYAN}{B}🎥 Lecture démarrée : {nom_source}{RS}")
    print(f"{GRY}   ESPACE = pause/reprise   |   Q = quitter{RS}\n")

    fenetre = "🚗 Plaques Belges — Vidéo"

    while True:
        # ── Gestion pause ──────────────────────────────────────
        if etat['pause']:
            frame_affichage = dessiner_frame(frame_pause, detections_courantes, etat)
            cv2.imshow(fenetre, frame_affichage)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                etat['pause'] = False
                print(f"{GRN}▶  Reprise{RS}")
            elif key in (ord('q'), ord('Q'), 27):
                break
            continue

        # ── Lecture frame ──────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            # Fin de fichier vidéo → reboucler
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # ── Envoi au worker OCR (chaque frame) ────────────────
        maintenant = time.time()
        if maintenant - derniere_envoi > 0.05:   # max ~20 fps vers l'OCR
            if not q_in.full():
                q_in.put(frame.copy())
                derniere_envoi = maintenant

        # ── Récupérer résultats OCR si disponibles ─────────────
        try:
            nouvelles_dets = q_out.get_nowait()
            detections_courantes = nouvelles_dets

            for det in nouvelles_dets:
                if det['valide']:
                    etat['derniere_valide'] = det
                    if det['texte'] not in etat['plaques_vues']:
                        etat['plaques_vues'].add(det['texte'])
                        etat['nb_valides'] += 1
                        ts = datetime.now().strftime("%H:%M:%S")
                        print(f"  {GRN}✅ [{ts}]  {B}{det['texte']}{RS}  "
                              f"(confiance {det['score']:.0%})")
        except queue.Empty:
            pass

        # ── Rendu ──────────────────────────────────────────────
        frame_affichage = dessiner_frame(frame, detections_courantes, etat)

        # Redimensionner si trop grande
        hf, wf = frame_affichage.shape[:2]
        if max(hf, wf) > 1280:
            s = 1280 / max(hf, wf)
            frame_affichage = cv2.resize(frame_affichage,
                                         (int(wf * s), int(hf * s)))

        cv2.imshow(fenetre, frame_affichage)

        # ── Touches clavier ────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            etat['pause']  = True
            frame_pause    = frame.copy()
            print(f"{YLW}⏸  Pause{RS}")
        elif key in (ord('q'), ord('Q'), 27):
            break

    # ── Nettoyage ──────────────────────────────────────────────
    worker.arreter()
    cap.release()
    cv2.destroyAllWindows()

    # ── Bilan terminal ─────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"{CYAN}{B}📊 Bilan de session{RS}")
    print(f"   Plaques uniques détectées : {etat['nb_valides']}")
    if etat['plaques_vues']:
        for p in sorted(etat['plaques_vues']):
            print(f"   {GRN}• {p}{RS}")
    print()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def banner():
    print(f"""
{CYAN}{B}╔══════════════════════════════════════════════════╗
║   🎥 Lecteur de Plaques Belges — Vidéo/Webcam   ║
╚══════════════════════════════════════════════════╝{RS}
""")


def main():
    banner()

    parser = argparse.ArgumentParser(
        description="Détection de plaques belges en temps réel.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    groupe = parser.add_mutually_exclusive_group(required=True)
    groupe.add_argument('--webcam', metavar='ID', nargs='?', const=0, type=int,
                        help='Webcam (défaut: 0, ou --webcam 1 pour la 2ème caméra)')
    groupe.add_argument('--video',  metavar='FICHIER',
                        help='Fichier vidéo (mp4, avi, mkv...)')
    args = parser.parse_args()

    # ── Ouverture source vidéo ─────────────────────────────────
    if args.webcam is not None:
        idx = args.webcam
        print(f"{CYAN}📷 Ouverture webcam #{idx}...{RS}")
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"{RED}[ERREUR] Webcam #{idx} introuvable.{RS}")
            print(f"{GRY}  Vérifiez que la caméra est connectée et non utilisée.{RS}")
            sys.exit(1)
        # Réglages webcam pour meilleure qualité
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        nom_source = f"Webcam #{idx}"

    else:
        chemin = args.video
        if not os.path.isfile(chemin):
            print(f"{RED}[ERREUR] Fichier introuvable : {chemin}{RS}")
            sys.exit(1)
        print(f"{CYAN}🎬 Ouverture vidéo : {chemin}{RS}")
        cap = cv2.VideoCapture(chemin)
        if not cap.isOpened():
            print(f"{RED}[ERREUR] Impossible d'ouvrir la vidéo.{RS}")
            sys.exit(1)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duree = total / fps if fps > 0 else 0
        print(f"   {total} frames  |  {fps:.1f} FPS  |  {duree:.1f}s\n")
        nom_source = os.path.basename(chemin)

    boucle_video(cap, nom_source, args)


if __name__ == '__main__':
    main()