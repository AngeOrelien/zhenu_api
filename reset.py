"""
reset.py
========
Réinitialise le projet EchoSign en supprimant les données générées.
Tu choisis exactement ce que tu veux effacer.

Usage :
    python reset.py              # mode interactif (recommandé)
    python reset.py --all        # tout effacer sans confirmation
    python reset.py --dataset    # dataset .npy uniquement
    python reset.py --models     # modèles entraînés uniquement
    python reset.py --logs       # logs TensorBoard uniquement
    python reset.py --raw        # vidéos/images sources uniquement
"""

import shutil
import argparse
from pathlib import Path

# ─── Chemins ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent

TARGETS = {
    "dataset": {
        "path":        ROOT / "dataset",
        "description": "Séquences .npy (dataset d'entraînement)",
        "warning":     "⚠️  IRRÉVERSIBLE — tu devras tout réenregistrer.",
    },
    "models": {
        "path":        ROOT / "ai_engine" / "models",
        "description": "Modèles entraînés (action.h5, labels.npy...)",
        "warning":     "⚠️  Tu devras réentraîner le modèle.",
    },
    "logs": {
        "path":        ROOT / "ai_engine" / "logs",
        "description": "Logs TensorBoard",
        "warning":     None,
    },
    "raw": {
        "path":        ROOT / "raw_data",
        "description": "Vidéos/images sources (raw_data/)",
        "warning":     "⚠️  Les fichiers sources seront supprimés définitivement.",
    },
}

# ─── Couleurs terminal ────────────────────────────────────────────────────────

RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def _folder_size(path: Path) -> str:
    """Calcule la taille d'un dossier de façon lisible."""
    if not path.exists():
        return "inexistant"
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    n_files = sum(1 for f in path.rglob("*") if f.is_file())
    if total < 1024:
        return f"{total} B  ({n_files} fichiers)"
    elif total < 1024 ** 2:
        return f"{total/1024:.1f} KB  ({n_files} fichiers)"
    elif total < 1024 ** 3:
        return f"{total/1024**2:.1f} MB  ({n_files} fichiers)"
    else:
        return f"{total/1024**3:.2f} GB  ({n_files} fichiers)"


def _delete(path: Path, label: str):
    """Supprime un dossier et le recrée vide."""
    if not path.exists():
        print(f"  {GRAY}⏭  {label} — dossier inexistant, ignoré.{RESET}")
        return
    shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    print(f"  {GREEN}✓  {label} supprimé et réinitialisé.{RESET}")


def _print_banner():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════╗
║         EchoSign Vision  —  Script de Reset      ║
╚══════════════════════════════════════════════════╝{RESET}
""")


def _print_status():
    """Affiche l'état actuel de chaque dossier."""
    print(f"{BOLD}  État actuel du projet :{RESET}")
    for key, info in TARGETS.items():
        size = _folder_size(info["path"])
        print(f"  {CYAN}{key:<10}{RESET}  {info['description']:<45}  {GRAY}{size}{RESET}")
    print()


# ─── Reset interactif ─────────────────────────────────────────────────────────

def run_interactive():
    _print_banner()
    _print_status()

    print(f"{BOLD}  Que veux-tu réinitialiser ?{RESET}")
    print()

    choices = {}
    for key, info in TARGETS.items():
        path = info["path"]
        exists = path.exists() and any(path.rglob("*"))

        if not exists:
            print(f"  {GRAY}[{key}]{RESET}  {info['description']}  {GRAY}— vide, ignoré{RESET}")
            choices[key] = False
            continue

        if info["warning"]:
            print(f"  {YELLOW}{info['warning']}{RESET}")

        answer = input(f"  Supprimer  [{CYAN}{key}{RESET}]  {info['description']} ? [o/N] > ").strip().lower()
        choices[key] = answer in ("o", "oui", "y", "yes")
        print()

    # Récapitulatif
    selected = [k for k, v in choices.items() if v]
    if not selected:
        print(f"  {GREEN}Rien n'a été supprimé.{RESET}\n")
        return

    print(f"\n{BOLD}  Récapitulatif — ce qui sera supprimé :{RESET}")
    for key in selected:
        print(f"  {RED}✗  {TARGETS[key]['description']}  ({TARGETS[key]['path']}){RESET}")

    print()
    confirm = input(f"  {BOLD}{RED}Confirmer la suppression ? [oui/N] > {RESET}").strip().lower()
    if confirm not in ("oui", "yes"):
        print(f"\n  {GREEN}Annulé — aucune modification.{RESET}\n")
        return

    print()
    for key in selected:
        _delete(TARGETS[key]["path"], TARGETS[key]["description"])

    print(f"\n  {GREEN}{BOLD}Reset terminé.{RESET}")
    _print_next_steps(selected)


# ─── Reset direct (flags CLI) ─────────────────────────────────────────────────

def run_direct(flags: dict):
    _print_banner()
    _print_status()

    selected = [k for k, v in flags.items() if v and k in TARGETS]

    if not selected:
        print(f"  {YELLOW}Aucun flag spécifié. Lance sans argument pour le mode interactif.{RESET}\n")
        return

    print(f"{BOLD}  {RED}Suppression en cours...{RESET}\n")
    for key in selected:
        _delete(TARGETS[key]["path"], TARGETS[key]["description"])

    print(f"\n  {GREEN}{BOLD}Reset terminé.{RESET}")
    _print_next_steps(selected)


# ─── Messages post-reset ──────────────────────────────────────────────────────

def _print_next_steps(deleted: list):
    print(f"\n{BOLD}  Prochaines étapes :{RESET}")

    if "dataset" in deleted:
        print(f"  {CYAN}→{RESET}  Recollecte le dataset :")
        print(f"         python ai_engine/webcam_collector.py")

    if "models" in deleted and "dataset" not in deleted:
        print(f"  {CYAN}→{RESET}  Réentraîne le modèle :")
        print(f"         python ai_engine/train_model.py")

    if "dataset" in deleted:
        print(f"  {CYAN}→{RESET}  Puis réentraîne le modèle :")
        print(f"         python ai_engine/train_model.py")

    if "raw" in deleted:
        print(f"  {CYAN}→{RESET}  Remets tes vidéos dans raw_data/<SIGNE>/")
        print(f"         puis : python ai_engine/data_processor.py")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EchoSign — Reset du projet",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--all",     action="store_true", help="Tout supprimer")
    parser.add_argument("--dataset", action="store_true", help="Supprimer les .npy")
    parser.add_argument("--models",  action="store_true", help="Supprimer les modèles")
    parser.add_argument("--logs",    action="store_true", help="Supprimer les logs")
    parser.add_argument("--raw",     action="store_true", help="Supprimer raw_data/")
    args = parser.parse_args()

    any_flag = args.all or args.dataset or args.models or args.logs or args.raw

    if not any_flag:
        run_interactive()
    else:
        run_direct({
            "dataset": args.all or args.dataset,
            "models":  args.all or args.models,
            "logs":    args.all or args.logs,
            "raw":     args.all or args.raw,
        })
