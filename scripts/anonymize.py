#!/usr/bin/env python3
"""Anonymize Polish legal / administrative text — fully local.

Detects PII with the lexedit HerBERT NER model (downloaded once from HuggingFace,
then runs offline via ONNX) + checksum-validated regex for structured IDs, then
replaces it with reversible ``[TYPE_HHHHHHHH]`` markers. The original text never
leaves the machine.

Deps:  pip install onnxruntime transformers huggingface_hub numpy

Usage:
  # anonymize a file (writes masked text + a reversible mapping)
  python anonymize.py --in contract.txt --out contract.masked.txt --map contract.map.json

  # anonymize inline text -> prints masked text; mapping to --map if given
  python anonymize.py --text "Pozwany Jan Kowalski, PESEL 02070803628."

  # restore the originals
  python anonymize.py --deanonymize --in contract.masked.txt --map contract.map.json --out contract.restored.txt

  # use the clean-text variant instead of the OCR-robust default
  python anonymize.py --in doc.txt --model lexedit/herbert-polish-legal-ner
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

DEFAULT_MODEL = "lexedit/herbert-polish-legal-ner-ocr"   # OCR-robust; good default for mixed docs
PER_THRESHOLD = 0.2                                       # recall-first: anonymisation wants recall >> precision
CHUNK, OVERLAP = 2000, 150

# Policy: which entity types get masked. Public places / dates / money / orgs stay
# visible so the legal context survives. Override with --mask / --keep.
DEFAULT_MASK = {"PER", "LOC", "ID", "IBAN", "EMAIL", "PHONE",
                "DIAGNOSIS", "HEALTH_FACILITY", "MEDICAL_ID",
                "PESEL", "NIP", "POSTAL"}

MARKER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*_[0-9A-F]{8}\]")


# --------------------------------------------------------------------------- model
def load_model(repo: str):
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    local = snapshot_download(repo, allow_patterns=["*.json", "*.txt", "onnx/*"])
    tok = AutoTokenizer.from_pretrained(local)
    cfg = json.load(open(Path(local) / "config.json", encoding="utf-8"))
    sess = ort.InferenceSession(str(Path(local) / "onnx" / "model_quantized.onnx"))
    return tok, cfg, sess, np


def ner_spans(text, tok, cfg, sess, np):
    """Return [{type,start,end}] over the whole doc (chunked, recall-first PER)."""
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    pb, pi = cfg["label2id"]["B-PER"], cfg["label2id"]["I-PER"]
    in_names = {i.name for i in sess.get_inputs()}
    out = []
    pos = 0
    while pos < len(text):
        chunk = text[pos:pos + CHUNK]
        enc = tok(chunk, return_offsets_mapping=True, return_tensors="np",
                  truncation=True, max_length=512)
        feeds = {"input_ids": enc["input_ids"].astype(np.int64),
                 "attention_mask": enc["attention_mask"].astype(np.int64)}
        if "token_type_ids" in in_names:
            feeds["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)
        logits = sess.run(None, feeds)[0][0]
        e = np.exp(logits - logits.max(-1, keepdims=True)); probs = e / e.sum(-1, keepdims=True)
        ids = probs.argmax(-1)
        cur = None
        for i, (s, en) in enumerate(enc["offset_mapping"][0]):
            if s == en:
                if cur: out.append(cur); cur = None
                continue
            lab = id2label[int(ids[i])]
            if lab == "O" and probs[i, pb] + probs[i, pi] >= PER_THRESHOLD:
                lab = "I-PER" if (cur and cur["type"] == "PER") else "B-PER"
            if lab == "O":
                if cur: out.append(cur); cur = None
                continue
            tag, et = lab.split("-", 1)
            gs, ge = pos + int(s), pos + int(en)
            if tag == "B" or cur is None or cur["type"] != et:
                if cur: out.append(cur)
                cur = {"type": et, "start": gs, "end": ge}
            else:
                cur["end"] = ge
        if cur: out.append(cur)
        if pos + CHUNK >= len(text): break
        pos += CHUNK - OVERLAP
    return out


# ---------------------------------------------------------------- structured regex
def luhn_like(digits, weights):
    return sum(int(d) * w for d, w in zip(digits, weights))


def valid_pesel(d):
    if len(d) != 11 or not d.isdigit(): return False
    chk = (10 - luhn_like(d[:10], [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]) % 10) % 10
    return chk == int(d[10])


def valid_nip(d):
    if len(d) != 10 or not d.isdigit(): return False
    c = luhn_like(d[:9], [6, 5, 7, 2, 3, 4, 5, 6, 7]) % 11
    return c != 10 and c == int(d[9])


def valid_iban(s):
    s = re.sub(r"\s", "", s).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s): return False
    r = s[4:] + s[:4]
    n = "".join(str(ord(c) - 55) if c.isalpha() else c for c in r)
    return int(n) % 97 == 1


def structured(text):
    out = []
    for m in re.finditer(r"(?<!\d)\d{11}(?!\d)", text):
        if valid_pesel(m.group()): out.append({"type": "PESEL", "start": m.start(), "end": m.end()})
    for m in re.finditer(r"(?<!\d)\d{3}-\d{3}-\d{2}-\d{2}(?!\d)|(?<!\d)\d{10}(?!\d)", text):
        if valid_nip(re.sub(r"\D", "", m.group())): out.append({"type": "NIP", "start": m.start(), "end": m.end()})
    for m in re.finditer(r"\b[A-Z]{2}\d{2}[ ]?(?:[A-Z0-9]{4}[ ]?){2,7}[A-Z0-9]{1,4}\b", text):
        if valid_iban(m.group()): out.append({"type": "IBAN", "start": m.start(), "end": m.end()})
    for m in re.finditer(r"(?<!\d)\d{2}(?:[ ]\d{4}){6}(?!\d)", text):           # bare PL IBAN, no country prefix
        if valid_iban("PL" + re.sub(r"\s", "", m.group())): out.append({"type": "IBAN", "start": m.start(), "end": m.end()})
    for m in re.finditer(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", text):
        out.append({"type": "EMAIL", "start": m.start(), "end": m.end()})
    for m in re.finditer(r"(?<!\d)(?:\+48[\s-]?)?\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)", text):
        out.append({"type": "PHONE", "start": m.start(), "end": m.end()})
    for m in re.finditer(r"(?<!\d)\d{2}-\d{3}(?!\d)", text):
        out.append({"type": "POSTAL", "start": m.start(), "end": m.end()})
    return out


# ------------------------------------------------------------------- post-process
def snap(text, spans):
    """Extend each span to whole orthographic words (kills subword-truncation tails)."""
    word = re.compile(r"[\w’'-]", re.UNICODE)
    for sp in spans:
        a, b = sp["start"], sp["end"]
        while a > 0 and word.match(text[a - 1]): a -= 1
        while b < len(text) and word.match(text[b]): b += 1
        sp["start"], sp["end"] = a, b
    return spans


def propagate_per(text, spans):
    """Mask other occurrences of a detected surname (exact token + >=5-char stem)."""
    extra = []
    surnames = set()
    for sp in spans:
        if sp["type"] == "PER":
            toks = [t for t in re.split(r"\s+", text[sp["start"]:sp["end"]]) if len(t) >= 4]
            if toks: surnames.add(toks[-1].strip(".,;:()"))
    for sur in surnames:
        stem = re.escape(sur[:max(5, len(sur) - 2)])
        for m in re.finditer(rf"\b{stem}[\wąćęłńóśżź]*", text, re.UNICODE | re.IGNORECASE):
            if m.group()[:1].isupper():
                extra.append({"type": "PER", "start": m.start(), "end": m.end()})
    return spans + extra


def merge(spans):
    spans = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))
    out = []
    for s in spans:
        if out and s["start"] < out[-1]["end"]:
            if (s["end"] - s["start"]) > (out[-1]["end"] - out[-1]["start"]):
                out[-1] = s
            continue
        out.append(s)
    return out


# ------------------------------------------------------------------------ masking
def marker(text, typ):
    h = hashlib.sha1(f"{typ}:{text}".encode("utf-8")).hexdigest()[:8].upper()
    return f"[{typ}_{h}]"


def anonymize(text, mask_types, model):
    tok, cfg, sess, np = model
    spans = ner_spans(text, tok, cfg, sess, np)
    spans = snap(text, spans)
    spans = propagate_per(text, spans)
    spans += structured(text)
    spans = [s for s in merge(spans) if s["type"] in mask_types]
    mapping = {}
    masked = text
    for s in sorted(spans, key=lambda x: x["start"], reverse=True):
        orig = text[s["start"]:s["end"]]
        mk = marker(orig, s["type"])
        mapping[mk] = orig
        masked = masked[:s["start"]] + mk + masked[s["end"]:]
    return masked, mapping, len(spans)


def deanonymize(masked, mapping):
    return MARKER_RE.sub(lambda m: mapping.get(m.group(), m.group()), masked)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Local Polish PII anonymizer (lexedit HerBERT)")
    ap.add_argument("--in", dest="inp", help="input text file (.txt/.md)")
    ap.add_argument("--text", help="inline text instead of a file")
    ap.add_argument("--out", help="write result here (else stdout)")
    ap.add_argument("--map", dest="mapping", help="reversible mapping JSON path")
    ap.add_argument("--deanonymize", action="store_true", help="restore originals (needs --map)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model id (default {DEFAULT_MODEL})")
    ap.add_argument("--mask", help="comma-separated types to mask (override default policy)")
    ap.add_argument("--keep", help="comma-separated types to KEEP visible (subtract from policy)")
    args = ap.parse_args()

    text = args.text if args.text is not None else (
        Path(args.inp).read_text(encoding="utf-8") if args.inp else sys.stdin.read())

    if args.deanonymize:
        if not args.mapping:
            ap.error("--deanonymize requires --map")
        mapping = json.load(open(args.mapping, encoding="utf-8"))
        result = deanonymize(text, mapping)
        n = len(MARKER_RE.findall(text))
        out = result
    else:
        mask_types = set(args.mask.split(",")) if args.mask else set(DEFAULT_MASK)
        if args.keep:
            mask_types -= set(args.keep.split(","))
        result, mapping, n = anonymize(text, mask_types, load_model(args.model))
        if args.mapping:
            json.dump(mapping, open(args.mapping, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        out = result

    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"{'restored' if args.deanonymize else 'masked'} {n} item(s) -> {args.out}"
              + (f"  (mapping -> {args.mapping})" if args.mapping and not args.deanonymize else ""), file=sys.stderr)
    else:
        sys.stdout.write(out)
        if not args.deanonymize:
            print(f"\n--- {n} item(s) masked"
                  + (f"; mapping -> {args.mapping}" if args.mapping else "; pass --map to save a reversible mapping")
                  + " ---", file=sys.stderr)


if __name__ == "__main__":
    main()
