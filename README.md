# lexedit-anonymizer — Claude Code skill

A [Claude Code](https://code.claude.com) **skill** that anonymizes / redacts
**personal data (PII) in Polish legal & administrative text** — fully **on your
machine**. It detects names (including inflected and OCR-garbled forms), private
addresses, PESEL / NIP / REGON / IBAN, emails, phones and medical data, masks them
with reversible markers, and can restore them later.

Powered by the open **HerBERT** Polish legal NER models on HuggingFace
([general](https://huggingface.co/lexedit/herbert-polish-legal-ner) ·
[OCR-robust](https://huggingface.co/lexedit/herbert-polish-legal-ner-ocr)) running
locally via ONNX. **No document text ever leaves the machine.**

```text
in:   Pozwany Jan Kowalski, zam. ul. Słoneczna 5 w Krakowie, PESEL 02070803628,
      e-mail jan.kowalski@example.pl. Pełnomocnik: r.pr. Anna Nowak-Kowalska.
out:  Pozwany [PER_6E06C0CA], zam. [LOC_A533FC9E] w Krakowie, PESEL [ID_D574D3B3],
      e-mail [EMAIL_8F6CC331]. Pełnomocnik: [PER_1F8FDEA2].
      # 'Krakowie' (public city) and the court are kept; reversible via the mapping
```

## Install

```bash
git clone https://github.com/tuul-ai/lexedit-anonymizer-skill ~/.claude/skills/lexedit-anonymizer
pip install -r ~/.claude/skills/lexedit-anonymizer/requirements.txt
```

Claude Code auto-discovers the skill (no restart). For a single project instead of
globally, clone into `<project>/.claude/skills/lexedit-anonymizer`. The model
(~125 MB) downloads from HuggingFace on first use, then runs offline.

## Use

Just ask Claude, e.g. *"zanonimizuj ten dokument"*, *"redact the PII in contract.txt
before I share it"*, *"scrub personal data from this ruling but keep a way to
reverse it"* — Claude invokes the skill automatically. Or run the engine directly:

```bash
# anonymize a file -> masked text + reversible mapping
python3 ~/.claude/skills/lexedit-anonymizer/scripts/anonymize.py \
  --in contract.txt --out contract.masked.txt --map contract.map.json

# restore the originals
python3 ~/.claude/skills/lexedit-anonymizer/scripts/anonymize.py \
  --deanonymize --in contract.masked.txt --map contract.map.json --out contract.txt

# clean-text variant (slightly more precise on digital docs)
... --model lexedit/herbert-polish-legal-ner
```

## What it masks

| Masked | Kept (legal context) |
|---|---|
| persons, private addresses, PESEL, NIP, REGON, IBAN, email, phone, postal code, medical (diagnosis / facility / id) | public places (city/country), organisations / courts, dates, amounts |

Override with `--mask TYPE1,TYPE2` or `--keep TYPE1,TYPE2`. Markers are
`[TYPE_HHHHHHHH]` (same entity → same marker); the `--map` JSON reverses them.

## How it works

Chunk → HerBERT NER with a recall-first PER threshold → snap spans to whole words →
propagate a detected surname to its other (inflected / OCR-variant) mentions →
checksum-validated regex for structured IDs (PESEL mod-10, NIP mod-11, IBAN mod-97)
→ reversible masking. See [`reference.md`](reference.md) and
[`SKILL.md`](SKILL.md).

## Privacy & limits

- **Local.** Everything runs via ONNX on your machine; the only network call is the
  one-time model download.
- **The `*.map.json` holds the real PII** — keep it secure; never commit or share it
  (the included `.gitignore` blocks it).
- **Strong first pass, not a guarantee.** Polish only; heavily OCR-garbled or
  out-of-distribution names can slip through. Review high-stakes output.

## License

Skill code: **MIT** (see [LICENSE](LICENSE)). The HerBERT models it downloads are
**CC BY 4.0** (attribution required, commercial use allowed) — please credit
**lexedit** and the base model `allegro/herbert-base-cased`.

> Polish legal NER / anonymisation by **lexedit** (https://lexedit.ai), models
> CC BY 4.0, based on HerBERT (`allegro/herbert-base-cased`).
