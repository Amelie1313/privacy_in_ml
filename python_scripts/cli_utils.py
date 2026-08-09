"""
Shared helper so every module prints section headers the same way,
(Original / Input / Output / Internal / Membership Attack / Row Removal /
Preprocessing)
"""

_WIDTH = 96
_RULE = "-" * _WIDTH


def print_banner(title):
    """Prints a centered, dashed-line banner for a *section* within a phase
    (e.g. "INPUT PERTURBATION", "MEMBERSHIP ATTACK - INPUT-PERTURBED MODELS")."""
    print(f"\n{_RULE}\n{title.center(_WIDTH)}\n{_RULE}\n")


def print_major_banner(title):
    """Prints a heavier '#'-ruled banner for a *phase* of main.py's run
    (PREPROCESSING / PERTURBATIONS / ROW REMOVAL / MEMBERSHIP ATTACK), one
    level up from print_banner's per-section dashes."""
    rule = "#" * _WIDTH
    print(f"\n{rule}\n{title.center(_WIDTH)}\n{rule}")
