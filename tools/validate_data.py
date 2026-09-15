#!/usr/bin/env python3
"""validate_data.py — sanity-check data/words.csv + data/sentences.csv, then write app/data.json.

Errors (exit code 1):
  * empty fields, non-integer / duplicate ids, block outside 1-7, unknown pos
  * the same `spanish` value twice inside one block (raw string; `bailás` and `¿bailás?` are distinct), or the same german+spanish pair anywhere
  * an infinitive stored with pos=verb (verbs are stored as spoken forms: tengo / tenés / tiene)
  * non-Rioplatense forms (aquí, dinero, autobús, vosotros, coche, ordenador, móvil, tú, tienes, ...)
    in words or sentences — the message names the porteño form
  * sentence word_ids that reference unknown ids, or whose form does not occur in the sentence
  * duplicate sentences, sentences without word_ids, words that occur in no sentence
Warnings:
  * sentence length outside 4-9 words (lyric lines are exempt), Spain/Mexico-flavoured vocabulary,
    the same spanish form in two different blocks (allowed when the meaning differs)

Usage: python tools/validate_data.py [--check-only] [--strict] [--quiet]
  --check-only  validate but do not write app/data.json
  --strict      treat warnings as errors
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS_CSV = ROOT / "data" / "words.csv"
SENTS_CSV = ROOT / "data" / "sentences.csv"
DATA_JSON = ROOT / "app" / "data.json"

POS = {"noun", "verb", "adj", "adv", "func", "phrase"}
BLOCKS = set(range(1, 8))
WORD_FIELDS = ["id", "block", "german", "spanish", "pos", "note"]
SENT_FIELDS = ["id", "word_ids", "spanish", "german"]  # + optional "note"

# token -> Rioplatense replacement. Any of these is an error.
BANNED = {
    "aquí": "acá", "dinero": "plata", "autobús": "colectivo", "vosotros": "ustedes", "vosotras": "ustedes",
    "coche": "auto", "ordenador": "computadora", "móvil": "celular", "tú": "vos",
    "tienes": "tenés", "quieres": "querés", "puedes": "podés",
    # more tú-forms that differ from the vos-forms
    "eres": "sos", "sabes": "sabés", "haces": "hacés", "dices": "decís", "vienes": "venís", "sales": "salís",
    "vuelves": "volvés", "entiendes": "entendés", "necesitas": "necesitás", "sientes": "sentís",
    "sigues": "seguís", "bailas": "bailás", "caminas": "caminás", "escuchas": "escuchás", "hablas": "hablás",
    "tomas": "tomás", "pagas": "pagás", "buscas": "buscás", "esperas": "esperás", "llegas": "llegás",
    "llevas": "llevás", "cruzas": "cruzás", "comes": "comés", "prefieres": "preferís", "duermes": "dormís",
    "pides": "pedís", "vives": "vivís", "piensas": "pensás", "trabajas": "trabajás", "conoces": "conocés",
    # vosotros-forms
    "sois": "son", "estáis": "están", "tenéis": "tienen", "queréis": "quieren", "podéis": "pueden",
    "habéis": "han", "vais": "van", "os": "les / se", "vuestro": "de ustedes", "vuestra": "de ustedes",
}
# Spain / Mexico flavoured vocabulary: warning with the porteño alternative.
SUSPICIOUS = {
    "vale": "dale / bueno", "guay": "copado / bárbaro", "mola": "está bueno", "coger": "agarrar / tomar",
    "zumo": "jugo", "patata": "papa", "ahorita": "ahora / ya", "chido": "copado", "órale": "dale",
    "platicar": "charlar", "carro": "auto", "conducir": "manejar", "gafas": "anteojos / lentes",
    "falda": "pollera", "chaqueta": "campera", "bolígrafo": "birome / lapicera", "fresa": "frutilla",
    "aguacate": "palta", "camarero": "mozo", "apartamento": "departamento", "piscina": "pileta",
    "cerillas": "fósforos", "chaval": "pibe", "currar": "laburar", "melocotón": "durazno",
    "judías": "porotos / chauchas", "miras": "mirás (falls Verb)", "compras": "comprás (falls Verb)",
    "paras": "parás (falls Verb)", "giras": "girás (falls Verb)", "sientas": "sentás (falls Verb)",
    "usted": "vos (im Alltag; usted nur sehr förmlich)",
}
# object/reflexive pronouns that may precede a verb form ("me gusta", "se llama")
CLITICS = {"me", "te", "se", "le", "les", "nos", "lo", "la", "los", "las", "no"}

PUNCT = re.compile(r"[¿?¡!.,;:\"'()\[\]{}«»“”‘’…—–/]+")


def norm(s: str) -> str:
    """Lowercase, NFC, punctuation -> space, collapse whitespace. Accents are kept."""
    s = unicodedata.normalize("NFC", s or "").lower()
    s = PUNCT.sub(" ", s)
    return " ".join(s.split())


def toks(s: str) -> list[str]:
    return norm(s).split()


def contains_seq(seq: list[str], sub: list[str]) -> bool:
    """True if `sub` occurs as a contiguous run inside `seq`."""
    if not sub or len(sub) > len(seq):
        return False
    return any(seq[i:i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


def looks_like_infinitive(form: str) -> bool:
    core = [t for t in toks(form) if t not in CLITICS]
    if not core:
        return False
    first = core[0]
    return first.endswith(("ar", "er", "ir")) and len(first) >= 2


class Report:
    def __init__(self, quiet: bool = False):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.quiet = quiet

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def dump(self):
        for w in self.warnings:
            print(f"WARN  {w}")
        for e in self.errors:
            print(f"ERROR {e}")


def read_csv(path: Path, required: list[str], rep: Report) -> list[dict]:
    if not path.exists():
        rep.error(f"{path.relative_to(ROOT)} fehlt")
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in required if c not in header]
        if missing:
            rep.error(f"{path.name}: Spalten fehlen im Header: {missing}")
            return []
        rows = []
        for n, row in enumerate(reader, start=2):
            row = {k: (v or "").strip() for k, v in row.items() if k is not None}
            row["_line"] = n
            rows.append(row)
        return rows


def check_tokens(text: str, where: str, rep: Report):
    for t in toks(text):
        if t in BANNED:
            rep.error(f"{where}: »{t}« ist nicht Rioplatense → »{BANNED[t]}«")
        elif t in SUSPICIOUS:
            rep.warn(f"{where}: »{t}« klingt nach Spanien/Mexiko → »{SUSPICIOUS[t]}«")


def validate_words(rows: list[dict], rep: Report) -> dict[int, dict]:
    words: dict[int, dict] = {}
    by_block_form: dict[tuple[int, str], int] = {}
    pairs: dict[tuple[str, str], int] = {}
    form_blocks: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        where = f"words.csv Zeile {r['_line']}"
        for k in ("id", "block", "german", "spanish", "pos"):
            if not r.get(k):
                rep.error(f"{where}: Feld »{k}« ist leer")
        try:
            wid = int(r["id"])
            if wid <= 0:
                raise ValueError
        except ValueError:
            rep.error(f"{where}: id »{r.get('id')}« ist keine positive Zahl")
            continue
        if wid in words:
            rep.error(f"{where}: id {wid} doppelt")
            continue
        try:
            block = int(r["block"])
        except ValueError:
            block = -1
        if block not in BLOCKS:
            rep.error(f"{where}: block »{r['block']}« nicht in 1-7")
        if r["pos"] not in POS:
            rep.error(f"{where}: pos »{r['pos']}« unbekannt (erlaubt: {sorted(POS)})")
        form = norm(r["spanish"])
        if not form:
            rep.error(f"{where}: spanish ist leer")
            continue
        raw_key = (block, " ".join(unicodedata.normalize("NFC", r["spanish"]).lower().split()))
        if raw_key in by_block_form:
            rep.error(f"{where}: »{r['spanish']}« kommt in Block {block} schon vor (id {by_block_form[raw_key]})")
        by_block_form[raw_key] = wid
        pair = (norm(r["german"]), form)
        if pair in pairs:
            rep.error(f"{where}: identisches Paar »{r['german']}« / »{r['spanish']}« wie id {pairs[pair]}")
        pairs[pair] = wid
        form_blocks[form].add(block)
        if r["pos"] == "verb" and looks_like_infinitive(r["spanish"]):
            rep.error(f"{where}: »{r['spanish']}« sieht nach Infinitiv aus – Verben als gesprochene Form speichern (yo / vos / él)")
        check_tokens(r["spanish"], where, rep)
        words[wid] = {
            "id": wid, "block": block, "german": r["german"], "spanish": r["spanish"],
            "pos": r["pos"], "note": r.get("note", ""), "_form": form, "_toks": form.split(),
        }
    for form, blocks in form_blocks.items():
        if len(blocks) > 1:
            rep.warn(f"»{form}« steht in mehreren Blöcken {sorted(blocks)} (ok, wenn andere Bedeutung)")
    return words


def validate_sentences(rows: list[dict], words: dict[int, dict], rep: Report) -> list[dict]:
    sents: list[dict] = []
    seen_ids: set[int] = set()
    seen_forms: dict[str, int] = {}
    covered: set[int] = set()
    for r in rows:
        where = f"sentences.csv Zeile {r['_line']}"
        for k in ("id", "word_ids", "spanish", "german"):
            if not r.get(k):
                rep.error(f"{where}: Feld »{k}« ist leer")
        try:
            sid = int(r["id"])
            if sid <= 0:
                raise ValueError
        except ValueError:
            rep.error(f"{where}: id »{r.get('id')}« ist keine positive Zahl")
            continue
        if sid in seen_ids:
            rep.error(f"{where}: id {sid} doppelt")
            continue
        seen_ids.add(sid)
        note = r.get("note", "") or ""
        stoks = toks(r["spanish"])
        sform = " ".join(stoks)
        if sform in seen_forms:
            rep.error(f"{where}: Satz identisch mit id {seen_forms[sform]}")
        seen_forms[sform] = sid
        ids: list[int] = []
        for part in [p for p in re.split(r"[;,\s]+", r["word_ids"]) if p]:
            try:
                wid = int(part)
            except ValueError:
                rep.error(f"{where}: word_id »{part}« ist keine Zahl")
                continue
            if wid not in words:
                rep.error(f"{where}: word_id {wid} existiert nicht")
                continue
            if not contains_seq(stoks, words[wid]["_toks"]):
                rep.error(f"{where}: Wort {wid} »{words[wid]['spanish']}« kommt im Satz nicht vor")
                continue
            if wid not in ids:
                ids.append(wid)
        if not ids:
            rep.error(f"{where}: keine gültigen word_ids")
        covered.update(ids)
        n = len(stoks)
        if note != "lyric" and not 4 <= n <= 9:
            rep.warn(f"{where}: {n} Wörter (Ziel 4-9): »{r['spanish']}«")
        check_tokens(r["spanish"], where, rep)
        sents.append({"id": sid, "word_ids": ids, "spanish": r["spanish"], "german": r["german"], "note": note})
    missing = [w for wid, w in sorted(words.items()) if wid not in covered]
    for w in missing:
        rep.error(f"Wort {w['id']} »{w['spanish']}« (Block {w['block']}) kommt in keinem Satz vor")
    return sents


def write_data_json(words: dict[int, dict], sents: list[dict]) -> str:
    payload = {
        "words": [{k: w[k] for k in ("id", "block", "german", "spanish", "pos", "note")} for _, w in sorted(words.items())],
        "sentences": sents,
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    version = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    payload = {"version": version, **payload}
    DATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    DATA_JSON.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return version


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true", help="nicht app/data.json schreiben")
    ap.add_argument("--strict", action="store_true", help="Warnungen wie Fehler behandeln")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    rep = Report(quiet=args.quiet)

    word_rows = read_csv(WORDS_CSV, WORD_FIELDS, rep)
    sent_rows = read_csv(SENTS_CSV, SENT_FIELDS, rep)
    words = validate_words(word_rows, rep) if word_rows else {}
    sents = validate_sentences(sent_rows, words, rep) if sent_rows and words else []

    if not args.quiet:
        rep.dump()
    per_block = defaultdict(int)
    for w in words.values():
        per_block[w["block"]] += 1
    blocks_str = " ".join(f"B{b}={per_block[b]}" for b in sorted(per_block))
    print(f"Wörter: {len(words)} ({blocks_str}) | Sätze: {len(sents)} | Fehler: {len(rep.errors)} | Warnungen: {len(rep.warnings)}")
    failed = bool(rep.errors) or (args.strict and rep.warnings)
    if failed:
        print("FEHLGESCHLAGEN")
        return 1
    if not args.check_only:
        v = write_data_json(words, sents)
        print(f"app/data.json geschrieben (version {v})")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
