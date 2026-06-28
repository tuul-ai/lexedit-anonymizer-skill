---
name: lexedit-anonymizer
description: >-
  Anonymize / redact personal data (PII) in POLISH legal & administrative text —
  person names (including inflected and OCR-garbled forms), private addresses,
  PESEL / NIP / REGON / IBAN, emails, phones, and medical data. Runs FULLY LOCAL
  (HerBERT NER via ONNX; no text leaves the machine) and produces a masked
  document plus a reversible mapping to de-anonymize later. Use when asked to
  anonymize, pseudonymize, redact, or scrub PII from a Polish document, contract,
  court ruling, email, or log — or to prepare a Polish document before sharing it
  or sending it to another LLM.
allowed-tools: Bash(python3 *) Bash(pip *)
---

# Polish PII anonymizer (lexedit / HerBERT, local)

Detects and masks personal data in **Polish** legal/administrative text with a
local NER model + checksum-validated regex, and can reverse the masking. **The
text never leaves the machine.**

## When to use

The user wants to remove / hide / pseudonymize personal data in Polish text:
"zanonimizuj", "usuń dane osobowe", "redact this contract", "scrub PII before I
share it", "prepare this ruling for an LLM", etc.

## Setup (first run only)

Dependencies (install once into the active environment):

```bash
pip install onnxruntime transformers huggingface_hub numpy
```

The first run downloads the model from HuggingFace (~125 MB, then cached and
offline). Default model `lexedit/herbert-polish-legal-ner-ocr` (OCR-robust).

## How to run

The engine is `scripts/anonymize.py`. Always call it with an absolute path:

**Anonymize a file** (writes masked text + a reversible mapping):
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/anonymize.py \
  --in input.txt --out input.masked.txt --map input.map.json
```

**Anonymize inline text** (prints masked text to stdout):
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/anonymize.py --text "Pozwany Jan Kowalski, PESEL 02070803628."
```

**Restore the originals** (reversible):
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/anonymize.py \
  --deanonymize --in input.masked.txt --map input.map.json --out input.restored.txt
```

**Choose the variant** — default is OCR-robust; for clean digital text the
general model is slightly more precise:
```bash
... --model lexedit/herbert-polish-legal-ner
```

**Adjust policy** (what gets masked):
```bash
... --keep ORG,DATE        # keep more visible
... --mask PER,ID,IBAN     # mask only these types
```

## Working with .docx / .pdf

The engine takes plain text. For Word/PDF: extract the text first (read the file,
or use a docx/pdf tool), pass it via `--in extracted.txt` or `--text`, then write
the masked result back into the document if needed.

## What it does

- Masks (by default): person names, private addresses, PESEL, NIP, REGON, IBAN,
  email, phone, postal code, and medical entities (diagnosis / facility / id).
- Keeps visible (legal context): public places (city/country), organisations /
  courts, dates, amounts.
- Output markers are `[TYPE_HHHHHHHH]` — the same entity gets the same marker, and
  the `--map` JSON lets you reverse it exactly.

For the full label scheme, masking policy, model details, and limitations, see
[reference.md](reference.md).

## Important

- **The `--map` file contains the original PII** — treat it as sensitive; do not
  share it alongside the masked document.
- This is a strong first pass, **not a guarantee**. Heavily OCR-garbled or
  out-of-distribution names can slip through. For high-stakes anonymisation, show
  the user the masked result + the mapping and recommend a human review.
- After anonymizing, briefly report to the user *what* was masked (counts per type
  from the mapping) so they can sanity-check coverage.
