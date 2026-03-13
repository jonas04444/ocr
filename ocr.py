#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════╗
║       Lecteur de lignes numériques sur image     ║
║       Utilise Tesseract OCR + OpenCV             ║
╚══════════════════════════════════════════════════╝

Usage :
    python lire_chiffres.py <chemin_image>
    python lire_chiffres.py <chemin_image> --save résultat.png
"""

import sys
import re
import argparse
import os

try:
    import pytesseract
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    import numpy as np
except ImportError as e:
    print(f"[ERREUR] Bibliothèque manquante : {e}")
    print("Installez les dépendances avec :")
    print("  pip install pytesseract pillow opencv-python")
    print("  Et Tesseract OCR : https://github.com/tesseract-ocr/tesseract")
    sys.exit(1)


# ─────────────────────────────────────────────────
#  Couleurs pour l'affichage terminal
# ─────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
GREY   = "\033[90m"





def pretraiter_image(image_pil):
    """
    Améliore la lisibilité de l'image pour Tesseract :
    - Conversion en niveaux de gris
    - Augmentation du contraste
    - Légère débruitage
    """
    img_cv = np.array(image_pil)

    # Conversion en niveaux de gris si nécessaire
    if len(img_cv.shape) == 3:
        gris = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    else:
        gris = img_cv

    # Débruitage léger
    gris = cv2.medianBlur(gris, 3)

    # Binarisation adaptative (gère les variations de luminosité)
    binaire = cv2.adaptiveThreshold(
        gris, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    return Image.fromarray(binaire)


def contient_chiffre(texte):
    """Retourne True si la chaîne contient au moins un chiffre."""
    return bool(re.search(r'\d', texte))


def extraire_lignes_numeriques(image_pil):
    """
    Extrait les lignes de texte contenant au moins un chiffre,
    avec leurs boîtes englobantes.

    Retourne :
        lignes       : liste de dict {texte, conf, x, y, w, h}
        texte_brut   : tout le texte détecté par Tesseract
    """
    # Config Tesseract : PSM 6 = bloc de texte uniforme
    config = r'--oem 3 --psm 6'

    # Extraction avec données de position
    donnees = pytesseract.image_to_data(
        image_pil,
        config=config,
        lang='fra+eng',
        output_type=pytesseract.Output.DICT
    )

    texte_brut = pytesseract.image_to_string(image_pil, config=config, lang='fra+eng')

    # Regroupement par numéro de ligne
    lignes_dict = {}
    n = len(donnees['text'])

    for i in range(n):
        mot = donnees['text'][i].strip()
        if not mot:
            continue

        conf = int(donnees['conf'][i])
        if conf < 0:  # Tesseract renvoie -1 pour les éléments de structure
            continue

        # Clé unique par bloc + paragraphe + ligne
        cle_ligne = (
            donnees['block_num'][i],
            donnees['par_num'][i],
            donnees['line_num'][i]
        )

        if cle_ligne not in lignes_dict:
            lignes_dict[cle_ligne] = {
                'mots': [],
                'confs': [],
                'x_min': donnees['left'][i],
                'y_min': donnees['top'][i],
                'x_max': donnees['left'][i] + donnees['width'][i],
                'y_max': donnees['top'][i] + donnees['height'][i],
            }

        lignes_dict[cle_ligne]['mots'].append(mot)
        lignes_dict[cle_ligne]['confs'].append(conf)
        lignes_dict[cle_ligne]['x_min'] = min(lignes_dict[cle_ligne]['x_min'], donnees['left'][i])
        lignes_dict[cle_ligne]['y_min'] = min(lignes_dict[cle_ligne]['y_min'], donnees['top'][i])
        lignes_dict[cle_ligne]['x_max'] = max(lignes_dict[cle_ligne]['x_max'],
                                               donnees['left'][i] + donnees['width'][i])
        lignes_dict[cle_ligne]['y_max'] = max(lignes_dict[cle_ligne]['y_max'],
                                               donnees['top'][i] + donnees['height'][i])

    # Filtrer les lignes avec au moins un chiffre
    lignes_numeriques = []
    for cle, info in lignes_dict.items():
        texte_ligne = ' '.join(info['mots'])
        if contient_chiffre(texte_ligne):
            conf_moy = sum(info['confs']) / len(info['confs']) if info['confs'] else 0
            lignes_numeriques.append({
                'texte': texte_ligne,
                'conf':  round(conf_moy, 1),
                'x':     info['x_min'],
                'y':     info['y_min'],
                'w':     info['x_max'] - info['x_min'],
                'h':     info['y_max'] - info['y_min'],
            })

    return lignes_numeriques, texte_brut

def afficher_resultats(lignes):

    if not lignes:
        print(f"{YELLOW}⚠  Aucune ligne numérique détectée.{RESET}")
        print(f"{GREY}   Conseils : vérifiez la qualité/résolution de l'image.{RESET}\n")
        return

    print(f"{GREEN}{BOLD}✅ {len(lignes)} ligne(s) numérique(s) détectée(s) :{RESET}\n")
    print(f"  {'N°':<4} {'Confiance':<11} {'Texte'}")
    print(f"  {'─'*4} {'─'*10} {'─'*50}")

    for i, ligne in enumerate(lignes, 1):
        conf = ligne['conf']
        couleur_conf = GREEN if conf >= 70 else (YELLOW if conf >= 40 else RED)
        print(f"  {i:<4} {couleur_conf}{conf:>6.1f} %{RESET}   {ligne['texte']}")

    print()


def annoter_image(image_pil, lignes):
    """
    Dessine des rectangles colorés autour des lignes numériques
    et affiche l'image annotée.
    """
    # Conversion en RGB pour OpenCV
    img_cv = cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

    for i, ligne in enumerate(lignes):
        x, y, w, h = ligne['x'], ligne['y'], ligne['w'], ligne['h']
        conf = ligne['conf']

        # Couleur selon la confiance (BGR)
        if conf >= 70:
            couleur = (50, 200, 50)    # Vert
        elif conf >= 40:
            couleur = (0, 180, 255)    # Orange
        else:
            couleur = (0, 50, 220)     # Rouge

        # Rectangle avec épaisseur selon la confiance
        epaisseur = 2 if conf >= 70 else 1
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), couleur, epaisseur)

        # Étiquette avec le numéro de ligne
        label = f"#{i+1} ({conf:.0f}%)"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

        # Fond de l'étiquette
        label_y = max(y - 5, lh + 5)
        cv2.rectangle(img_cv,
                      (x, label_y - lh - 4),
                      (x + lw + 6, label_y + 2),
                      couleur, -1)
        cv2.putText(img_cv, label,
                    (x + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return img_cv


def sauvegarder_image(img_cv, chemin_sortie):
    """Sauvegarde l'image annotée."""
    cv2.imwrite(chemin_sortie, img_cv)
    print(f"{GREEN}💾 Image annotée sauvegardée : {chemin_sortie}{RESET}\n")


def afficher_image(img_cv, titre="Lignes numériques détectées", chemin_fallback="resultat_annote.png"):
    """
    Affiche l'image dans une fenêtre OpenCV.
    Si aucun écran n'est disponible (serveur, SSH...), sauvegarde automatiquement.
    """
    # Redimensionner si trop grande pour l'écran
    h, w = img_cv.shape[:2]
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))

    try:
        print(f"{CYAN}📷 Affichage de l'image annotée...{RESET}")
        print(f"{GREY}   (Appuyez sur une touche pour fermer){RESET}\n")
        cv2.imshow(titre, img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        # Pas d'écran disponible (serveur, SSH sans X11...) → sauvegarde auto
        print(f"{YELLOW}⚠  Aucun écran disponible pour l'affichage.{RESET}")
        cv2.imwrite(chemin_fallback, img_cv)
        print(f"{GREEN}💾 Image annotée sauvegardée automatiquement : {chemin_fallback}{RESET}\n")
        print(f"{GREY}   Ouvrez ce fichier pour voir les zones détectées.{RESET}\n")


# ─────────────────────────────────────────────────
#  Point d'entrée principal
# ─────────────────────────────────────────────────
def main():

    parser = argparse.ArgumentParser(
        description="Détecte et affiche les lignes contenant des chiffres dans une image.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('image',
                        help='Chemin vers l\'image (jpg, jpeg, png, bmp, tiff)')
    parser.add_argument('--save', metavar='FICHIER',
                        help='Sauvegarder l\'image annotée (ex: résultat.png)')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Désactiver le prétraitement de l\'image')

    args = parser.parse_args()

    # ── Vérification du fichier ──────────────────
    if not os.path.isfile(args.image):
        print(f"{RED}[ERREUR] Fichier introuvable : {args.image}{RESET}")
        sys.exit(1)

    ext = os.path.splitext(args.image)[1].lower()
    if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'):
        print(f"{YELLOW}[ATTENTION] Extension non standard : {ext}{RESET}")

    # ── Chargement de l'image ────────────────────
    print(f"{CYAN}📂 Chargement de l'image : {args.image}{RESET}")
    try:
        image_originale = Image.open(args.image)
    except Exception as e:
        print(f"{RED}[ERREUR] Impossible d'ouvrir l'image : {e}{RESET}")
        sys.exit(1)

    print(f"   Taille : {image_originale.size[0]} × {image_originale.size[1]} px\n")

    # ── Prétraitement ────────────────────────────
    if not args.no_preprocess:
        print(f"{CYAN}⚙  Prétraitement de l'image...{RESET}")
        image_traitee = pretraiter_image(image_originale)
    else:
        image_traitee = image_originale

    # ── OCR ─────────────────────────────────────
    print(f"{CYAN}🔎 Analyse OCR en cours...{RESET}\n")
    try:
        lignes, _ = extraire_lignes_numeriques(image_traitee)
    except pytesseract.TesseractNotFoundError:
        print(f"{RED}[ERREUR] Tesseract OCR n'est pas installé ou introuvable.{RESET}")
        print("  Téléchargez-le sur : https://github.com/tesseract-ocr/tesseract")
        sys.exit(1)

    # ── Affichage des résultats ──────────────────
    afficher_resultats(lignes)

    # ── Annotation et affichage de l'image ──────
    if lignes:
        img_annotee = annoter_image(image_originale, lignes)

        if args.save:
            sauvegarder_image(img_annotee, args.save)

        fallback = args.save if args.save else "resultat_annote.png"
        afficher_image(img_annotee, chemin_fallback=fallback)
    else:
        print(f"{GREY}Aucune annotation à afficher.{RESET}\n")


if __name__ == '__main__':
    main()