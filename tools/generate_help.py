#!/usr/bin/env python3
"""Genera il contenuto HTML del manuale utente e lo inietta nelle pagine.

Legge docs/MANUALE_UTENTE.md, lo converte in HTML e sostituisce il blocco
compreso tra i marcatori HELP_MANUAL_START / HELP_MANUAL_END in
app/cashrec.html.

In questo modo il file markdown resta l'unica fonte del manuale: dopo ogni
modifica basta rieseguire:  python3 tools/generate_help.py
"""
import re
from pathlib import Path

try:
    import markdown as md
except ImportError:
    md = None

ROOT = Path(__file__).resolve().parent.parent
MANUAL = ROOT / "docs" / "MANUALE_UTENTE.md"
TARGETS = [ROOT / "app" / "cashrec.html"]

START = "<!-- HELP_MANUAL_START -->"
END = "<!-- HELP_MANUAL_END -->"


def render(md_text):
    if md is None:
        raise SystemExit(
            "Il pacchetto 'markdown' non è installato (pip install markdown)."
        )
    return md.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )


def inject(path, html):
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(
            f"Marcatori help non trovati in {path}: integrare prima "
            f"pulsante + modal + script, poi rieseguire."
        )
    replacement = START + "\n" + html + "\n" + END
    new_text, n = pattern.subn(replacement, text, count=1)
    if n != 1:
        raise SystemExit(f"Nessuna sostituzione eseguita in {path}")
    path.write_text(new_text, encoding="utf-8")
    print(f"Aggiornato: {path.relative_to(ROOT)}")


def main():
    md_text = MANUAL.read_text(encoding="utf-8")
    html = render(md_text)
    for target in TARGETS:
        inject(target, html)
    print("Fatto.")


if __name__ == "__main__":
    main()