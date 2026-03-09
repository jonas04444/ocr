#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║    🚗 Lecteur de Plaques Belges — Version Optimisée  ║
║    Format : 1-ABC-234  |  Photos distance / angle    ║
╚══════════════════════════════════════════════════════╝

Usage :
    python plaque_belge.py <image>
    python plaque_belge.py <image> --save resultat.png
    python plaque_belge.py <image> --debug        ← affiche chaque étape
"""

import sys, re, os, argparse
import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    print("[ERREUR] pip install pytesseract pillow opencv-python")
    sys.exit(1)

# ── Couleurs terminal ──────────────────────────────────────────
R="\033[0m"; B="\033[1m"; CYAN="\033[96m"; GRN="\033[92m"
YLW="\033[93m"; RED="\033[91m"; GRY="\033[90m"

# ── Format plaque belge ────────────────────────────────────────
# Nouvelle : 1-ABC-234   (depuis 2010)
# Ancienne : ABC-123     (avant 2010, encore en circulation)
PATTERN_NOUVELLE = re.compile(r'^[0-9]-[A-Z]{3}-[0-9]{3}$')
PATTERN_ANCIENNE = re.compile(r'^[A-Z]{1,3}-[0-9]{1,4}-[A-Z]{0,3}$')

# ── Corrections caractères courants ────────────────────────────
# Position connue dans la plaque → forcer lettre ou chiffre
CORRECTIONS_OCR = {
    # Chiffres souvent confondus avec des lettres
    'O': '0', 'o': '0', 'Q': '0',
    'I': '1', 'l': '1', '|': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'G': '6', 'b': '6',
    'B': '8',
    'g': '9',
}

def banner():
    print(f"""
{CYAN}{B}╔══════════════════════════════════════════════╗
║  🚗 Lecteur de Plaques Belges — Optimisé    ║
╚══════════════════════════════════════════════╝{R}
""")


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 1 — Détection de la zone plaque
# ══════════════════════════════════════════════════════════════
def detecter_zone_plaque(img_bgr, debug=False):
    """
    Localise la plaque dans l'image via :
      1. Filtre de couleur (blanc / jaune belge)
      2. Détection de contours rectangulaires
      3. Score de confiance (ratio largeur/hauteur ~4.5)
    Retourne la liste des ROI candidates, triées par score.
    """
    h_img, w_img = img_bgr.shape[:2]
    candidates = []

    # ── Méthode A : filtre couleur blanche (plaques récentes) ──
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    masque_blanc = cv2.inRange(hsv,
                               np.array([0,   0, 180]),
                               np.array([180, 40, 255]))

    # ── Méthode B : filtre couleur jaune (plaques particuliers) ─
    masque_jaune = cv2.inRange(hsv,
                                np.array([18, 60, 100]),
                                np.array([35, 255, 255]))

    for nom, masque in [("blanc", masque_blanc), ("jaune", masque_jaune)]:
        # Morphologie pour remplir les trous
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, kernel)
        masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            aire = cv2.contourArea(c)
            if aire < 1000:   # trop petit
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw   # toujours largeur > hauteur

            if rh < 5:
                continue
            ratio = rw / rh
            # Plaque belge ≈ 520×113 mm → ratio ≈ 4.6
            score_ratio = 1.0 - abs(ratio - 4.6) / 4.6
            if score_ratio < 0.2:
                continue

            # Score surface relative
            score_aire = min(aire / (w_img * h_img * 0.15), 1.0)

            score = score_ratio * 0.7 + score_aire * 0.3
            x, y, w, hh = cv2.boundingRect(c)
            candidates.append({
                'x': x, 'y': y, 'w': w, 'h': hh,
                'score': score, 'methode': nom, 'angle': angle
            })

        if debug:
            _debug_show(f"Masque couleur ({nom})", masque)

    # ── Méthode C : gradient (fonctionne mieux la nuit / distance) ─
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    grad_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = np.uint8(np.clip(grad / grad.max() * 255, 0, 255))
    _, seuil = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    seuil = cv2.morphologyEx(seuil, cv2.MORPH_CLOSE, kernel_h)
    contours2, _ = cv2.findContours(seuil, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
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
        candidates.append({
            'x': x, 'y': y, 'w': w, 'h': hh,
            'score': score_ratio * 0.5, 'methode': 'gradient', 'angle': 0
        })

    # Trier par score décroissant
    candidates.sort(key=lambda c: -c['score'])

    if debug:
        _debug_show("Gradient seuillé", seuil)

    return candidates[:5]   # garder les 5 meilleures

# ══════════════════════════════════════════════════════════════
#  ÉTAPE 2 — Correction de perspective / angle
# ══════════════════════════════════════════════════════════════
def corriger_perspective(img_bgr, x, y, w, h, marge=8):
    """
    Redresse la plaque si elle est prise en angle.
    Utilise la transformation de perspective de OpenCV.
    """
    # Ajouter une marge autour de la ROI
    x1 = max(0, x - marge)
    y1 = max(0, y - marge)
    x2 = min(img_bgr.shape[1], x + w + marge)
    y2 = min(img_bgr.shape[0], y + h + marge)
    roi = img_bgr[y1:y2, x1:x2]

    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bin_ = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return roi

    # Chercher le plus grand contour quadrilatéral
    c_max = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(c_max, True)
    poly = cv2.approxPolyDP(c_max, epsilon, True)

    if len(poly) == 4:
        pts = poly.reshape(4, 2).astype(np.float32)
        # Ordonner : haut-gauche, haut-droite, bas-droite, bas-gauche
        pts_ord = _ordonner_points(pts)
        (tl, tr, br, bl) = pts_ord
        larg = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        haut = int(max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br)))
        dst = np.array([[0,0],[larg-1,0],[larg-1,haut-1],[0,haut-1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts_ord, dst)
        roi = cv2.warpPerspective(roi, M, (larg, haut))

    return roi


def _ordonner_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # haut-gauche
    rect[2] = pts[np.argmax(s)]   # bas-droite
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # haut-droite
    rect[3] = pts[np.argmax(diff)] # bas-gauche
    return rect


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 3 — Prétraitement avancé pour OCR plaque
# ══════════════════════════════════════════════════════════════
def pretraiter_plaque(roi_bgr, debug=False):
    """
    Pipeline de prétraitement optimisé pour les plaques :
      - Upscaling ×3 (Tesseract travaille mieux ≥ 300px de haut)
      - Débruitage
      - Contraste CLAHE
      - Binarisation Otsu
      - Érosion légère pour séparer les caractères collés
    Retourne plusieurs variantes (Tesseract vote sur la meilleure).
    """
    # Upscaling ×3 interpolation cubique
    h, w = roi_bgr.shape[:2]
    roi_bgr = cv2.resize(roi_bgr, (w * 3, h * 3),
                         interpolation=cv2.INTER_CUBIC)

    gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Débruitage non-local means (conserve mieux les bords)
    gris = cv2.fastNlMeansDenoising(gris, h=10, templateWindowSize=7,
                                    searchWindowSize=21)

    # CLAHE — améliore le contraste local (utile si éclairage inégal)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gris_clahe = clahe.apply(gris)

    variantes = {}

    # V1 : Otsu simple
    _, v1 = cv2.threshold(gris_clahe, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variantes['otsu'] = v1

    # V2 : Otsu inversé (plaque foncée sur fond clair)
    variantes['otsu_inv'] = cv2.bitwise_not(v1)

    # V3 : Seuillage adaptatif (meilleur si éclairage non uniforme)
    v3 = cv2.adaptiveThreshold(gris_clahe, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 19, 9)
    variantes['adaptatif'] = v3

    # V4 : Morphologie pour nettoyer
    kernel = np.ones((2, 2), np.uint8)
    v4 = cv2.morphologyEx(v1, cv2.MORPH_OPEN, kernel)
    variantes['morph'] = v4

    if debug:
        for nom, img in variantes.items():
            _debug_show(f"Prétraitement : {nom}", img)

    return variantes


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 4 — OCR + corrections caractères
# ══════════════════════════════════════════════════════════════

# Config Tesseract pour plaque sur une ligne
# PSM 7 = ligne unique  |  PSM 8 = mot unique
# Whitelist : uniquement les caractères autorisés dans une plaque belge
TESS_CONFIGS = [
    r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
    r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
    r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
]

def ocr_plaque(variantes: dict) -> list[dict]:
    """
    Lance Tesseract sur chaque variante d'image avec plusieurs configs.
    Retourne tous les résultats uniques triés par score de validité.
    """
    resultats = {}

    for nom_img, img in variantes.items():
        # Ajouter une bordure blanche (Tesseract aime les marges)
        img_m = cv2.copyMakeBorder(img, 20, 20, 20, 20,
                                   cv2.BORDER_CONSTANT, value=255)
        for cfg in TESS_CONFIGS:
            try:
                texte = pytesseract.image_to_string(
                    img_m, config=cfg, lang='eng'
                ).strip().upper()

                # Nettoyer : garder uniquement alphanum et tirets
                texte = re.sub(r'[^A-Z0-9\-]', '', texte)
                if not texte:
                    continue

                texte_corr = corriger_caracteres(texte)
                score = scorer_plaque(texte_corr)

                cle = texte_corr
                if cle not in resultats or score > resultats[cle]['score']:
                    resultats[cle] = {
                        'texte':     texte_corr,
                        'brut':      texte,
                        'score':     score,
                        'valide':    est_valide(texte_corr),
                        'variante':  f"{nom_img}",
                    }
            except Exception:
                continue

    return sorted(resultats.values(), key=lambda r: -r['score'])


def corriger_caracteres(texte: str) -> str:
    """
    Corrige les confusions OCR en tenant compte de la POSITION
    dans le format belge 1-ABC-234 :
      pos 0  → chiffre obligatoire
      pos 2-4 → lettres obligatoires
      pos 6-8 → chiffres obligatoires
    """
    # Normaliser les tirets (OCR lit parfois _ ou espace)
    texte = re.sub(r'[\s_–—]', '-', texte)
    # Supprimer les tirets en trop en début/fin
    texte = texte.strip('-')

    # Correction position par position si format complet
    if len(texte) == 9 and texte[1] == '-' and texte[5] == '-':
        chars = list(texte)
        # Position 0 : doit être un chiffre
        chars[0] = _forcer_chiffre(chars[0])
        # Positions 2,3,4 : doivent être des lettres
        for i in (2, 3, 4):
            chars[i] = _forcer_lettre(chars[i])
        # Positions 6,7,8 : doivent être des chiffres
        for i in (6, 7, 8):
            chars[i] = _forcer_chiffre(chars[i])
        return ''.join(chars)

    # Sinon : correction générique
    return ''.join(CORRECTIONS_OCR.get(c, c) for c in texte)


def _forcer_chiffre(c: str) -> str:
    """Convertit un caractère en chiffre si c'est une confusion connue."""
    MAP = {'O':'0','Q':'0','D':'0','I':'1','L':'1','l':'1',
           'Z':'2','S':'5','G':'6','B':'8','g':'9','A':'4'}
    return MAP.get(c, c) if not c.isdigit() else c


def _forcer_lettre(c: str) -> str:
    """Convertit un caractère en lettre si c'est une confusion connue."""
    MAP = {'0':'O','1':'I','5':'S','6':'G','8':'B','4':'A'}
    return MAP.get(c, c) if not c.isalpha() else c


def scorer_plaque(texte: str) -> float:
    """
    Score entre 0 et 1 mesurant la probabilité que le texte
    soit une plaque belge valide.
    """
    if not texte:
        return 0.0

    score = 0.0

    # Format nouvelle plaque (max score)
    if PATTERN_NOUVELLE.match(texte):
        return 1.0

    # Format ancienne plaque
    if PATTERN_ANCIENNE.match(texte):
        return 0.85

    # Longueur correcte
    if len(texte) == 9:
        score += 0.3
    elif len(texte) == 7:
        score += 0.2

    # Tirets aux bonnes positions
    if len(texte) > 1 and texte[1] == '-':
        score += 0.2
    if len(texte) > 5 and texte[5] == '-':
        score += 0.2

    # Contient au moins un chiffre et une lettre
    if re.search(r'\d', texte):
        score += 0.1
    if re.search(r'[A-Z]', texte):
        score += 0.1

    return min(score, 0.99)


def est_valide(texte: str) -> bool:
    return bool(PATTERN_NOUVELLE.match(texte) or PATTERN_ANCIENNE.match(texte))


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 5 — Annotation et affichage
# ══════════════════════════════════════════════════════════════
def annoter_image(img_bgr, candidates, resultats_par_candidate):
    """Dessine les zones détectées et les textes reconnus."""
    out = img_bgr.copy()

    for i, cand in enumerate(candidates):
        res_list = resultats_par_candidate.get(i, [])
        meilleur = res_list[0] if res_list else None

        x, y, w, h = cand['x'], cand['y'], cand['w'], cand['h']
        couleur = (0, 200, 0) if (meilleur and meilleur['valide']) else (0, 165, 255)

        cv2.rectangle(out, (x, y), (x + w, y + h), couleur, 2)

        label = meilleur['texte'] if meilleur else '?'
        score_txt = f" ({meilleur['score']:.0%})" if meilleur else ''

        # Fond du label
        (lw, lh), _ = cv2.getTextSize(label + score_txt,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ly = max(y - 8, lh + 8)
        cv2.rectangle(out, (x, ly - lh - 6), (x + lw + 8, ly + 4),
                      couleur, -1)
        cv2.putText(out, label + score_txt, (x + 4, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return out


def afficher(img_cv, titre="Résultat", chemin_fallback="resultat_plaque.png"):
    h, w = img_cv.shape[:2]
    if max(h, w) > 1100:
        s = 1100 / max(h, w)
        img_cv = cv2.resize(img_cv, (int(w*s), int(h*s)))
    try:
        cv2.imshow(titre, img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        cv2.imwrite(chemin_fallback, img_cv)
        print(f"{YLW}⚠  Pas d'écran → image sauvegardée : {chemin_fallback}{R}")


def _debug_show(titre, img):
    try:
        cv2.imshow(titre, img)
        cv2.waitKey(500)
    except cv2.error:
        pass


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    banner()

    parser = argparse.ArgumentParser(description="Lecteur de plaques belges optimisé")
    parser.add_argument('image', help='Chemin image (jpg/png)')
    parser.add_argument('--save',  metavar='FICHIER', help='Sauvegarder l\'image annotée')
    parser.add_argument('--debug', action='store_true', help='Afficher chaque étape de traitement')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"{RED}[ERREUR] Fichier introuvable : {args.image}{R}")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print(f"{RED}[ERREUR] Impossible de lire l'image.{R}")
        sys.exit(1)

    print(f"{CYAN}📂 Image chargée : {args.image}  ({img.shape[1]}×{img.shape[0]} px){R}\n")

    # ── 1. Détection des zones plaque ─────────────────────────
    print(f"{CYAN}🔍 Détection des zones plaque...{R}")
    candidates = detecter_zone_plaque(img, debug=args.debug)

    if not candidates:
        print(f"{YLW}⚠  Aucune zone plaque trouvée. Tentative OCR sur image complète.{R}")
        candidates = [{'x': 0, 'y': 0, 'w': img.shape[1],
                       'h': img.shape[0], 'score': 0.1, 'methode': 'full', 'angle': 0}]
    else:
        print(f"   {len(candidates)} zone(s) candidate(s) détectée(s).\n")

    # ── 2-4. Correction + prétraitement + OCR par candidate ───
    resultats_par_candidate = {}
    tous_resultats = []

    for i, cand in enumerate(candidates):
        print(f"{CYAN}🔧 Traitement zone #{i+1}  "
              f"[méthode: {cand['methode']}, score: {cand['score']:.2f}]{R}")

        # Correction de perspective
        roi = corriger_perspective(img, cand['x'], cand['y'],
                                   cand['w'], cand['h'])
        if args.debug:
            _debug_show(f"ROI corrigée #{i+1}", roi)

        # Prétraitement
        variantes = pretraiter_plaque(roi, debug=args.debug)

        # OCR
        resultats = ocr_plaque(variantes)
        resultats_par_candidate[i] = resultats

        for r in resultats[:3]:
            valide_txt = f"{GRN}✅ VALIDE{R}" if r['valide'] else f"{YLW}~{R}"
            print(f"   {valide_txt}  {B}{r['texte']}{R}  "
                  f"(confiance: {r['score']:.0%})")
            tous_resultats.append(r)

        print()

    # ── Résultat final ─────────────────────────────────────────
    valides = [r for r in tous_resultats if r['valide']]
    valides.sort(key=lambda r: -r['score'])

    print(f"{'─'*50}")
    if valides:
        best = valides[0]
        print(f"\n{GRN}{B}🏆 Plaque détectée : {best['texte']}{R}  "
              f"(confiance {best['score']:.0%})\n")
    else:
        non_valides = sorted(tous_resultats, key=lambda r: -r['score'])
        if non_valides:
            best = non_valides[0]
            print(f"\n{YLW}⚠  Meilleure tentative (format non standard) : "
                  f"{B}{best['texte']}{R}  ({best['score']:.0%})\n")
            print(f"{GRY}   Conseils si le résultat est mauvais :{R}")
            print(f"{GRY}   • Utilisez une image plus proche / mieux éclairée{R}")
            print(f"{GRY}   • Essayez --debug pour voir les étapes{R}")
        else:
            print(f"\n{RED}❌ Aucun texte lisible détecté.{R}\n")

    # ── Annotation image ───────────────────────────────────────
    img_annotee = annoter_image(img, candidates, resultats_par_candidate)

    chemin_sortie = args.save or "resultat_plaque.png"
    if args.save:
        cv2.imwrite(chemin_sortie, img_annotee)
        print(f"{GRN}💾 Image annotée sauvegardée : {chemin_sortie}{R}\n")

    afficher(img_annotee, chemin_fallback=chemin_sortie)


if __name__ == '__main__':
    main()