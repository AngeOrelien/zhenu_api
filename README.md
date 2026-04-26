# ZHẼNÙ API 🤟

API FastAPI de reconnaissance de langue des signes en temps réel.  
Reçoit des keypoints depuis l'app **ZHẼNÙ** (Flutter), prédit le signe via un modèle BiLSTM.

---

## Table des matières

1. [Structure du projet](#1-structure-du-projet)
2. [Convention de nommage des signes](#2-convention-de-nommage-des-signes)
3. [Installation sur Windows 10](#3-installation-sur-windows-10)
4. [Comprendre les fichiers .npy](#4-comprendre-les-fichiers-npy)
5. [Construire le dataset](#5-construire-le-dataset)
   - [Méthode A — Webcam (recommandée)](#méthode-a--webcam-recommandée)
   - [Méthode B — Vidéos et images existantes](#méthode-b--vidéos-et-images-existantes)
6. [Entraîner le modèle](#6-entraîner-le-modèle)
7. [Lancer le serveur API](#7-lancer-le-serveur-api)
8. [Tester sans Flutter](#8-tester-sans-flutter)
9. [Intégration Flutter](#9-intégration-flutter)
10. [Partager le dataset avec une collègue](#10-partager-le-dataset-avec-une-collègue)
11. [Réinitialiser le projet](#11-réinitialiser-le-projet)
12. [Paramètres importants](#12-paramètres-importants)
13. [Nombre de séquences recommandé](#13-nombre-de-séquences-recommandé)

---

## 1. Structure du projet

```
zhenu_api/
│
├── ai_engine/                   # Tout ce qui concerne l'IA
│   ├── webcam_collector.py      # ★ Collecte manuelle via webcam
│   ├── data_processor.py        # Conversion vidéos/images → .npy
│   ├── train_model.py           # Entraînement du modèle BiLSTM
│   ├── models/
│   │   ├── action.h5            # Modèle entraîné  (généré par train)
│   │   ├── action_best.h5       # Meilleur checkpoint (généré par train)
│   │   └── labels.npy           # Liste des signes dans l'ordre
│   └── logs/                    # Logs TensorBoard (générés par train)
│
├── app/                         # Serveur FastAPI
│   ├── main.py                  # Routes WebSocket + REST
│   ├── inference.py             # Chargement modèle + prédiction
│   └── processor.py             # Validation des keypoints reçus
│
├── dataset/                     # ★ Fichiers .npy — seul fichier nécessaire pour entraîner
│   ├── BONJOUR/
│   │   ├── 0.npy                # Une séquence = tableau numpy (30, 1692)
│   │   ├── 1.npy
│   │   └── ...
│   ├── AU_REVOIR/
│   └── ...
│
├── raw_data/                    # Vidéos/images brutes → non versionnées sur Git
├── mp_models/                   # Modèles MediaPipe Tasks → téléchargés automatiquement
│   ├── hand_landmarker.task
│   ├── pose_landmarker.task
│   └── face_landmarker.task
│
├── reset.py                     # Script de réinitialisation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 2. Convention de nommage des signes

> **Règle absolue : `MAJUSCULES_AVEC_UNDERSCORE`, sans accents, sans espaces**  
> Encore plus important sur Windows où les accents dans les chemins causent des erreurs.

| ✅ Correct        | ❌ Incorrect       | Raison                              |
|------------------|--------------------|-------------------------------------|
| `BONJOUR`        | `bonjour`          | Toujours majuscules                  |
| `AU_REVOIR`      | `au revoir`        | Espaces → erreurs de chemin          |
| `JE_TAIME`       | `je t'aime`        | Apostrophe → casse le terminal       |
| `COMMENT_CA_VA`  | `comment ça va`    | Accents → encoding Windows (cp1252)  |
| `A`              | `À`                | Accent → problème sur Windows        |

Le script `webcam_collector.py` nettoie automatiquement ce que tu tapes.

---

## 3. Installation sur Windows 10

### Prérequis
- Python 3.10, 3.11 ou 3.13 — télécharger sur [python.org](https://www.python.org/downloads/)
- Git — télécharger sur [git-scm.com](https://git-scm.com/)
- Cocher **"Add Python to PATH"** lors de l'installation

### Étapes

```bat
:: 1. Cloner le repo
git clone https://github.com/ton-compte/zhenu_api.git
cd zhenu_api

:: 2. Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate

:: 3. Installer les dépendances
pip install -r requirements.txt
```

### Ouvrir le terminal correctement sur Windows

Pour éviter les problèmes d'encodage avec Python sur Windows 10, utilise **PowerShell** ou **Git Bash** plutôt que CMD. Si tu utilises CMD, ajoute cette ligne en haut de chaque script Python :

```python
# En haut de chaque script si tu utilises CMD
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

Ou règle l'encodage une fois pour toutes dans ton terminal CMD :
```bat
chcp 65001
```

### Webcam sur Windows

Si la webcam ne s'ouvre pas, essaie l'index `1` au lieu de `0` dans `webcam_collector.py` :
```python
cap = cv2.VideoCapture(1)   # essaie 0, 1 ou 2 selon ta machine
```

---

## 4. Comprendre les fichiers .npy

Chaque fichier `.npy` représente **un geste filmé une fois**.

```
shape : (30, 1692)
         │    └── 1692 coordonnées extraites par MediaPipe par frame
         └────── 30 frames consécutives capturées
```

Composition des 1692 valeurs par frame :

```
Pose    : 33 points × 4 valeurs (x, y, z, visibility) =  132
Visage  : 478 points × 3 valeurs (x, y, z)             = 1434
Main G  : 21 points × 3 valeurs (x, y, z)              =   63
Main D  : 21 points × 3 valeurs (x, y, z)              =   63
                                               Total  = 1692
```

> **Important :** Les `.npy` sont les seuls fichiers nécessaires pour entraîner.  
> `train_model.py` lit directement `dataset/<SIGNE>/*.npy` — pas besoin des vidéos sources.

---

## 5. Construire le dataset

### Méthode A — Webcam (recommandée)

```bat
:: Mode interactif — le script te pose les questions
python ai_engine\webcam_collector.py

:: Mode direct — tout en arguments
python ai_engine\webcam_collector.py --sign BONJOUR --reps 30

:: Avec une longueur de séquence différente
python ai_engine\webcam_collector.py --sign BONJOUR --reps 30 --sequence_length 45
```

**Déroulement :**
1. Saisir le nom du signe dans le terminal (`BONJOUR`)
2. Saisir le nombre d'enregistrements (`30`)
3. La webcam s'ouvre — le signe est affiché en grand
4. Appuyer sur `ESPACE` quand tu es prêt → enregistrement automatique
5. Répété pour chaque enregistrement

**Touches :**

| Touche   | Action                            |
|----------|-----------------------------------|
| `ESPACE` | Démarrer l'enregistrement         |
| `M`      | Retour au menu (changer de signe) |
| `Q`      | Quitter                           |

---

### Méthode B — Vidéos et images existantes

Place tes fichiers dans `raw_data\` :

```
raw_data\
├── BONJOUR\
│   ├── clip1.mp4
│   └── photo.jpg
└── MERCI\
    └── merci.mp4
```

| Type   | Extensions acceptées                |
|--------|-------------------------------------|
| Vidéo  | `.mp4` `.avi` `.mov` `.mkv` `.webm` |
| Image  | `.jpg` `.jpeg` `.png` `.bmp` `.webp`|

```bat
python ai_engine\data_processor.py
python ai_engine\data_processor.py --sequence_length 45
```

---

**Vérifier l'état du dataset :**

```bat
python -c "from pathlib import Path; [print(f'{p.name:<25} {len(list(p.glob(chr(42)+\".npy\"))):>3} seq{\" ⚠\" if len(list(p.glob(chr(42)+\".npy\"))) < 20 else \"\"}') for p in sorted(Path('dataset').iterdir()) if p.is_dir()]"
```

---

## 6. Entraîner le modèle

> Le modèle lit uniquement les fichiers `.npy` dans `dataset\`.

```bat
python ai_engine\train_model.py

:: Options
python ai_engine\train_model.py --epochs 150
python ai_engine\train_model.py --batch_size 16
python ai_engine\train_model.py --no_plot
```

**Fichiers générés :**

```
ai_engine\models\
├── action.h5            ← modèle final
├── action_best.h5       ← meilleur checkpoint (utilisé par l'API)
├── labels.npy           ← liste des signes dans l'ordre
└── training_curves.png
```

---

## 7. Lancer le serveur API

```bat
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

| URL                                  | Description               |
|--------------------------------------|---------------------------|
| `http://localhost:8000/docs`         | Documentation interactive |
| `http://localhost:8000/health`       | Statut du serveur         |
| `http://localhost:8000/signs`        | Signes reconnus           |
| `ws://localhost:8000/ws/recognize`   | WebSocket temps réel      |

**Autoriser le port 8000 dans le pare-feu Windows (pour accès depuis Flutter physique) :**
```bat
netsh advfirewall firewall add rule name="ZHENU API" dir=in action=allow protocol=TCP localport=8000
```

---

## 8. Tester sans Flutter

Crée `test_websocket.py` à la racine :

```python
import asyncio, websockets, json, numpy as np

async def test():
    uri = "ws://localhost:8000/ws/recognize"
    async with websockets.connect(uri) as ws:
        for i in range(35):
            fake_keypoints = np.zeros(1692).tolist()
            await ws.send(json.dumps({"keypoints": fake_keypoints}))
            response = json.loads(await ws.recv())
            print(f"Frame {i+1}: {response}")

asyncio.run(test())
```

```bat
pip install websockets
python test_websocket.py
```

---

## 9. Intégration Flutter

```dart
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';

class ZhenuService {
  final channel = WebSocketChannel.connect(
    Uri.parse('ws://TON_IP:8000/ws/recognize'),
  );

  void sendFrame(List<double> keypoints) {
    channel.sink.add(jsonEncode({"keypoints": keypoints}));
  }

  void reset() {
    channel.sink.add(jsonEncode({"reset": true}));
  }

  Stream get results => channel.stream.map((msg) => jsonDecode(msg));
  void dispose() => channel.sink.close();
}
```

**Adresse IP selon le contexte :**

| Contexte                       | URL                                      |
|--------------------------------|------------------------------------------|
| Émulateur Android              | `ws://10.0.2.2:8000/ws/recognize`        |
| Appareil physique (même Wi-Fi) | `ws://192.168.X.X:8000/ws/recognize`     |
| Production                     | `wss://ton-domaine.com/ws/recognize`     |

Trouver ton IP locale sur Windows : `ipconfig` → cherche **Adresse IPv4**

---

## 10. Partager le dataset avec une collègue

### Pourquoi les `.npy` suffisent

> Ta collègue n'a besoin que des fichiers `.npy` pour entraîner le modèle.  
> Pas besoin des vidéos, pas besoin de MediaPipe ni de recollecte.  
> Elle clone, récupère les `.npy`, et lance `train_model.py` directement.

---

### Stratégie — deux branches Git

```
branche main     →  code uniquement  (jamais de .npy)
branche dataset  →  .npy uniquement  (jamais de code)
```

### Mise en place (toi — une seule fois)

```bat
:: Créer la branche dataset
git checkout -b dataset
git rm -r --cached .
git add dataset/
git commit -m "dataset: initialisation"
git push origin dataset

:: Retourner sur main
git checkout main
```

### Ajouter de nouvelles séquences

```bat
:: Enregistrer sur main
python ai_engine\webcam_collector.py

:: Mettre à jour la branche dataset
git checkout dataset
git checkout main -- dataset/
git add dataset/
git commit -m "dataset: ajout BONJOUR x30"
git push origin dataset
git checkout main
```

### Récupérer le dataset (ta collègue)

```bat
git clone https://github.com/ton-compte/zhenu_api.git
cd zhenu_api
git fetch origin
git checkout origin/dataset -- dataset/
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python ai_engine\train_model.py
```

---

## 11. Réinitialiser le projet

```bat
:: Mode interactif — te demande confirmation pour chaque élément
python reset.py

:: Modes directs
python reset.py --dataset     :: supprimer les .npy
python reset.py --models      :: supprimer les modèles .h5
python reset.py --logs        :: supprimer les logs
python reset.py --raw         :: supprimer les vidéos sources
python reset.py --all         :: tout supprimer
```

---

## 12. Paramètres importants

| Paramètre              | Valeur | Fichiers à modifier                                               |
|------------------------|--------|-------------------------------------------------------------------|
| `SEQUENCE_LENGTH`      | `30`   | `webcam_collector.py` + `train_model.py` + `inference.py`         |
| `KEYPOINTS_DIM`        | `1692` | `train_model.py` + `inference.py` + `processor.py`               |
| `CONFIDENCE_THRESHOLD` | `0.70` | `inference.py`                                                    |
| `MIN_SEQUENCES`        | `20`   | Recommandation minimale par signe                                 |

> ⚠️ Si tu changes `SEQUENCE_LENGTH`, **tout le dataset devient incompatible**.  
> Il faut réenregistrer toutes les séquences et réentraîner.

---

## 13. Nombre de séquences recommandé

Le nombre nécessaire est **proportionnel au nombre de signes**.

| Phase           | Séquences / signe | 10 signes     | 30 signes      |
|-----------------|-------------------|---------------|----------------|
| Test / débogage | 10 – 20           | 100 – 200     | 300 – 600      |
| Modèle correct  | 30 – 50           | 300 – 500     | 900 – 1500     |
| Production      | 80 – 150          | 800 – 1500    | 2400 – 4500    |

Des gestes proches visuellement nécessitent plus de séquences que des gestes très distincts.