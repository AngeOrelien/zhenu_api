"""
webcam_collector.py
===================
Collecte manuelle de séquences via la webcam pour construire le dataset.
Tu choisis le signe, le nombre d'enregistrements, puis tu fais le geste.
Les séquences sont sauvegardées en .npy dans dataset/<SIGNE>/.

Usage :
    python ai_engine/webcam_collector.py
    python ai_engine/webcam_collector.py --sign BONJOUR --reps 30
    python ai_engine/webcam_collector.py --sequence_length 45
"""

import cv2
import numpy as np
import argparse
import urllib.request
import time
from pathlib import Path

# ─── Chemins ──────────────────────────────────────────────────────────────────

_ROOT       = Path(__file__).parent.parent
_MODELS_DIR = _ROOT / "mp_models"
_DATASET    = _ROOT / "dataset"
_MODELS_DIR.mkdir(exist_ok=True)
_DATASET.mkdir(exist_ok=True)

# ─── Téléchargement modèles MediaPipe ─────────────────────────────────────────

_MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
}

def _ensure_models():
    for fname, url in _MODEL_URLS.items():
        dest = _MODELS_DIR / fname
        if not dest.exists():
            print(f"  ⬇  Téléchargement {fname} ...")
            urllib.request.urlretrieve(url, str(dest))
            print(f"  ✓  {fname} prêt.")

_ensure_models()

# ─── Import MediaPipe Tasks ───────────────────────────────────────────────────

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# ─── Constantes ───────────────────────────────────────────────────────────────

KEYPOINTS_DIM       = 1692   # pose(132) + face(1434) + mains(126)
DEFAULT_SEQ_LENGTH  = 30     # frames par séquence
DEFAULT_REPS        = 5     # répétitions par défaut
COUNTDOWN_SEC       = 3      # secondes avant enregistrement

# Couleurs BGR
GREEN  = (0, 220, 80)
RED    = (0, 50, 220)
YELLOW = (0, 210, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
PURPLE = (200, 50, 200)
GRAY   = (140, 140, 140)

# ─── Détecteur MediaPipe ──────────────────────────────────────────────────────

class _Detector:
    def __init__(self):
        Base = mp_python.BaseOptions
        self._pose = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=Base(model_asset_path=str(_MODELS_DIR / "pose_landmarker.task")),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._face = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=Base(model_asset_path=str(_MODELS_DIR / "face_landmarker.task")),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                num_faces=1,
            )
        )
        self._hands = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=Base(model_asset_path=str(_MODELS_DIR / "hand_landmarker.task")),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                num_hands=2,
            )
        )

    def detect(self, frame_bgr):
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_res = self._pose.detect(mp_img)
        face_res = self._face.detect(mp_img)
        hand_res = self._hands.detect(mp_img)

        pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
        face_lms = face_res.face_landmarks[0] if face_res.face_landmarks else None
        left_h = right_h = None
        if hand_res.hand_landmarks:
            for lm_list, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
                if handedness[0].category_name == "Left":
                    left_h  = lm_list
                else:
                    right_h = lm_list
        return pose_lms, face_lms, left_h, right_h

    def close(self):
        self._pose.close(); self._face.close(); self._hands.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def _extract_keypoints(pose, face, lh, rh) -> np.ndarray:
    p = (np.array([[lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.0)]
                   for lm in pose]).flatten() if pose else np.zeros(33 * 4))
    f_raw = (np.array([[lm.x, lm.y, lm.z] for lm in face]).flatten()
             if face else np.zeros(478 * 3))
    f = np.zeros(478 * 3)
    f[:min(len(f_raw), 478 * 3)] = f_raw[:478 * 3]
    l = (np.array([[lm.x, lm.y, lm.z] for lm in lh]).flatten()
         if lh else np.zeros(21 * 3))
    r = (np.array([[lm.x, lm.y, lm.z] for lm in rh]).flatten()
         if rh else np.zeros(21 * 3))
    return np.concatenate([p, f, l, r])


def _draw_skeleton(image, pose, lh, rh):
    h, w = image.shape[:2]
    def pt(lm): return int(lm.x * w), int(lm.y * h)

    POSE_CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),
                 (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)]
    HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                 (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

    if pose:
        for a, b in POSE_CONN:
            if a < len(pose) and b < len(pose):
                cv2.line(image, pt(pose[a]), pt(pose[b]), (80, 44, 121), 2)
        for lm in pose:
            cv2.circle(image, pt(lm), 3, (80, 22, 10), -1)

    for lms, dc, lc in [(lh, (121,22,76), (121,44,250)), (rh, (245,117,66), (245,66,230))]:
        if lms:
            for a, b in HAND_CONN:
                cv2.line(image, pt(lms[a]), pt(lms[b]), lc, 2)
            for lm in lms:
                cv2.circle(image, pt(lm), 4, dc, -1)

# ─── Affichage UI ─────────────────────────────────────────────────────────────

def _header(img, text, color=PURPLE):
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, 40), (w, 88), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.7, img, 0.3, 0, img)
    cv2.putText(img, text, (w//2 - len(text)*7, 72),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

def _progress_bar(img, current, total, label=""):
    h, w = img.shape[:2]
    by, bh, bw = h - 45, 18, w - 40
    cv2.rectangle(img, (20, by), (20+bw, by+bh), (60,60,60), -1)
    fill = int(bw * current / max(total, 1))
    cv2.rectangle(img, (20, by), (20+fill, by+bh), GREEN, -1)
    cv2.putText(img, f"{label}  {current}/{total}", (25, by+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

def _detection_status(img, pose, face, lh, rh):
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, 38), (25, 25, 25), -1)
    cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
    for label, detected, x in [
        ("POSE", pose is not None, 10),
        ("VISAGE", face is not None, 120),
        ("MAIN G", lh is not None, 250),
        ("MAIN D", rh is not None, 385),
    ]:
        cv2.circle(img, (x+10, 19), 7, GREEN if detected else RED, -1)
        cv2.putText(img, label, (x+22, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)

def _wait_screen(img, sign, rep_idx, total_reps):
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
    cv2.putText(img, "SIGNE A EFFECTUER :", (w//2 - 160, h//2 - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    cv2.putText(img, sign, (w//2 - len(sign)*16, h//2),
                cv2.FONT_HERSHEY_DUPLEX, 2.0, YELLOW, 3)
    cv2.putText(img, f"Enregistrement  {rep_idx+1} / {total_reps}",
                (w//2 - 140, h//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GRAY, 1)
    cv2.putText(img, "Appuie sur  [ESPACE]  quand tu es pret",
                (w//2 - 230, h//2 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREEN, 2)
    cv2.putText(img, "[Q] Quitter    [M] Changer de signe",
                (w//2 - 160, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

def _rec_indicator(img, frame_idx, seq_length):
    h, w = img.shape[:2]
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(img, (w - 30, 60), 10, RED, -1)
        cv2.putText(img, "REC", (w - 70, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    cv2.putText(img, f"Frame {frame_idx+1}/{seq_length}",
                (w - 185, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

# ─── Saisie du signe dans le terminal ─────────────────────────────────────────

def _ask_sign() -> str:
    print("\n" + "─"*50)
    print("  Entrez le nom du signe à enregistrer.")
    print("  Convention : MAJUSCULES + UNDERSCORE")
    print("  Exemples   : BONJOUR  /  JE_TAIME  /  AU_REVOIR  /  A")
    print("─"*50)
    while True:
        raw = input("  Signe > ").strip()
        if not raw:
            print("  ⚠  Nom vide, recommence.")
            continue
        # Nettoyage automatique
        clean = raw.upper().replace(" ", "_")
        for ch in "àâäéèêëîïôùûüç":
            clean = clean.replace(ch.upper(), ch.upper()
                        .replace("À","A").replace("Â","A")
                        .replace("Ä","A").replace("É","E")
                        .replace("È","E").replace("Ê","E")
                        .replace("Ë","E").replace("Î","I")
                        .replace("Ï","I").replace("Ô","O")
                        .replace("Ù","U").replace("Û","U")
                        .replace("Ü","U").replace("Ç","C"))
        # Garder seulement lettres, chiffres, underscore
        clean = "".join(c for c in clean if c.isalnum() or c == "_")
        if not clean:
            print("  ⚠  Nom invalide après nettoyage, recommence.")
            continue
        print(f"  ✓  Signe : {clean}")
        return clean


def _ask_reps(default: int) -> int:
    print(f"  Nombre d'enregistrements [{default}] > ", end="")
    raw = input().strip()
    if not raw:
        return default
    try:
        n = int(raw)
        if n < 1:
            raise ValueError
        return n
    except ValueError:
        print(f"  ⚠  Valeur invalide, utilisation du défaut ({default})")
        return default


# ─── Collecte d'un signe ──────────────────────────────────────────────────────

def collect_sign(
    cap,
    detector: _Detector,
    sign: str,
    total_reps: int,
    seq_length: int,
    start_rep: int = 0,
) -> str:
    """
    Enregistre `total_reps` séquences pour `sign`.
    Retourne 'done', 'menu' ou 'quit'.
    """
    out_dir = _DATASET / sign
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  ▶  Signe : {sign}  |  {total_reps - start_rep} enregistrement(s) à faire")

    for rep_idx in range(start_rep, total_reps):
        sequence = []

        # ── Écran d'attente : l'utilisateur appuie sur ESPACE ──────────────
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            pose, face, lh, rh = detector.detect(frame)
            _draw_skeleton(frame, pose, lh, rh)
            _detection_status(frame, pose, face, lh, rh)
            _wait_screen(frame, sign, rep_idx, total_reps)
            _progress_bar(frame, rep_idx, total_reps, "Enregistrements")
            cv2.imshow("EchoSign — Collecte Webcam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return 'quit'
            if key == ord('m'):
                return 'menu'
            if key == ord(' '):
                waiting = False

        # ── Enregistrement des frames ───────────────────────────────────────
        for frame_idx in range(seq_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            pose, face, lh, rh = detector.detect(frame)
            _draw_skeleton(frame, pose, lh, rh)
            _detection_status(frame, pose, face, lh, rh)
            _header(frame, f"Signe : {sign}", GREEN)
            _rec_indicator(frame, frame_idx, seq_length)
            _progress_bar(frame, rep_idx, total_reps, "Enregistrements")
            cv2.imshow("EchoSign — Collecte Webcam", frame)

            sequence.append(_extract_keypoints(pose, face, lh, rh))
            cv2.waitKey(1)

        # ── Sauvegarde ─────────────────────────────────────────────────────
        # Index = nombre de fichiers existants pour ne pas écraser
        existing = len(list(out_dir.glob("*.npy")))
        save_path = out_dir / f"{existing}.npy"
        np.save(str(save_path), np.array(sequence))
        print(f"  ✓  Rep {rep_idx+1:02d}/{total_reps}  →  {save_path.name}"
              f"  shape={np.array(sequence).shape}")

    return 'done'


# ─── Résumé du dataset ────────────────────────────────────────────────────────

def _print_summary():
    print("\n" + "="*55)
    print("  RÉSUMÉ DU DATASET")
    print("="*55)
    sign_dirs = sorted(_DATASET.iterdir()) if _DATASET.exists() else []
    if not sign_dirs:
        print("  (dataset vide)")
    for d in sign_dirs:
        if d.is_dir():
            n   = len(list(d.glob("*.npy")))
            bar = "█" * min(n, 40)
            print(f"  {d.name:<22} {bar:<40} {n}")
    print("="*55)


# ─── Boucle principale ────────────────────────────────────────────────────────

def run(sign_arg: str = None, reps_arg: int = None, seq_length: int = DEFAULT_SEQ_LENGTH):
    print("\n" + "="*55)
    print("  EchoSign Vision  —  Collecte Webcam")
    print("="*55)
    print("  Touches :")
    print("    [ESPACE]  Démarrer l'enregistrement")
    print("    [M]       Changer de signe (retour menu)")
    print("    [Q]       Quitter")
    print("="*55)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("❌ Webcam inaccessible.")
        return

    with _Detector() as detector:
        while True:
            # ── Choix du signe ──────────────────────────────────────────────
            if sign_arg:
                sign     = sign_arg.upper()
                sign_arg = None   # Une seule fois via CLI
            else:
                sign = _ask_sign()

            # ── Nombre d'enregistrements ────────────────────────────────────
            if reps_arg:
                reps     = reps_arg
                reps_arg = None
            else:
                reps = _ask_reps(DEFAULT_REPS)

            # ── Enregistrements existants pour ce signe ─────────────────────
            existing_count = len(list((_DATASET / sign).glob("*.npy"))) \
                             if (_DATASET / sign).exists() else 0
            if existing_count > 0:
                print(f"  ℹ  {existing_count} séquence(s) déjà enregistrée(s) pour {sign}.")
                print(f"     Les nouvelles seront ajoutées (index {existing_count}+).")

            # ── Collecte ────────────────────────────────────────────────────
            result = collect_sign(cap, detector, sign, reps, seq_length, start_rep=0)

            if result == 'quit':
                print("\n  ⛔ Collecte interrompue.")
                break
            elif result == 'done':
                print(f"\n  ✅ '{sign}' terminé — {reps} séquence(s) enregistrée(s).")
                again = input("  Continuer avec un autre signe ? [O/n] > ").strip().lower()
                if again == 'n':
                    break
            # 'menu' → retour au début de la boucle

    cap.release()
    cv2.destroyAllWindows()
    _print_summary()
    print("\n  Lance l'entraînement avec :")
    print("  python ai_engine/train_model.py\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EchoSign — Collecte webcam de séquences .npy"
    )
    parser.add_argument("--sign",            default=None,
                        help="Signe à enregistrer (ex: BONJOUR)")
    parser.add_argument("--reps",            type=int, default=None,
                        help=f"Nombre d'enregistrements (défaut: {DEFAULT_REPS})")
    parser.add_argument("--sequence_length", type=int, default=DEFAULT_SEQ_LENGTH,
                        help=f"Frames par séquence (défaut: {DEFAULT_SEQ_LENGTH})")
    args = parser.parse_args()

    run(sign_arg=args.sign, reps_arg=args.reps, seq_length=args.sequence_length)
