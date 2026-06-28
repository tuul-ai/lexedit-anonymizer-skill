# Reference — lexedit-anonymizer

## Model

- **`lexedit/herbert-polish-legal-ner-ocr`** (default) — OCR-robust; best for
  scanned / mixed documents. https://huggingface.co/lexedit/herbert-polish-legal-ner-ocr
- **`lexedit/herbert-polish-legal-ner`** — general; slightly higher precision on
  clean digital text. https://huggingface.co/lexedit/herbert-polish-legal-ner

Both are HerBERT (Polish BERT) token-classification models, int8 ONNX (~125 MB),
licensed **CC BY 4.0** (attribution required, commercial use allowed). Identity-level
person-name leak on an internal eval: ~7% (clean) / ~15% (scanned) with the
post-pass. Token-level F1 ≈ 0.94.

## Label scheme (29 labels, BIO)

| Type | Meaning | Default action |
|---|---|---|
| `PER` | person name (any inflected form) | **mask** |
| `LOC` | private address / location | **mask** |
| `LOC_PUB` | public place (city, country) | keep |
| `ORG` | organisation / court | keep |
| `ID` | national id / case / document number | **mask** |
| `IBAN` | bank account | **mask** |
| `EMAIL`, `PHONE` | contact handles | **mask** |
| `DIAGNOSIS`, `HEALTH_FACILITY`, `MEDICAL_ID` | medical PII | **mask** |
| `DATE`, `MONEY` | dates / amounts | keep |
| `WATERMARK` | document watermark | keep |

Structured IDs added by checksum-validated regex (independent of the model):
`PESEL` (mod-10), `NIP` (mod-11), `IBAN` (mod-97, incl. bare PL form), `EMAIL`,
`PHONE`, `POSTAL`.

Override with `--mask TYPE1,TYPE2` (mask only these) or `--keep TYPE1,TYPE2`
(remove from the default mask set).

## How it works (pipeline)

1. **Chunk** the text (2000 chars, 150 overlap) to fit the 512-token window.
2. **NER** per chunk with a **recall-first PER threshold** (flip a token to PER when
   summed PER probability ≥ 0.2, even if it is not the arg-max — anonymisation wants
   recall ≫ precision).
3. **Snap** every span to whole words (kills subword-truncation tails like
   `[PER]owskiej`).
4. **Propagate** a detected surname to its other occurrences (exact token + ≥5-char
   stem, capitalised) across the document.
5. **Structured regex** for IDs with checksum validation (precision gate).
6. **Merge** overlaps, **mask** with reversible `[TYPE_HHHHHHHH]` markers, emit the
   `original → marker` mapping.

## Limitations

- **Polish only.**
- **Not a guarantee** — heavily OCR-garbled names, foreign/out-of-distribution
  names, and OCR-mangled IDs can slip through. See the model cards'
  `KNOWN_LIMITATIONS.md` for concrete failure cases.
- The `--map` JSON holds the real PII — keep it secure.
- This is a first pass; combine with **human review** for high-stakes use.

## Privacy

Everything runs locally (ONNX). The only network call is the one-time model
download from HuggingFace; after that it works fully offline. No document text is
sent anywhere.
