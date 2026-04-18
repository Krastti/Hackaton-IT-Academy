import re
from typing import Dict, Set, List, Any, Optional, Iterable
import sys

try:
    import pymorphy3  # type: ignore
except ImportError:  # pragma: no cover - зависит от окружения
    pymorphy3 = None  # type: ignore
else:
    sys.modules["pymorphy2"] = pymorphy3
    sys.modules["pymorphy2.analyzer"] = pymorphy3.analyzer

try:
    from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger, Doc  # type: ignore
except ImportError:  # pragma: no cover - зависит от окружения
    Segmenter = None  # type: ignore
    MorphVocab = None  # type: ignore
    NewsEmbedding = None  # type: ignore
    NewsNERTagger = None  # type: ignore
    Doc = None  # type: ignore


def is_luhn_valid(card_number: str) -> bool:
    digits_list: List[int] = [int(d) for d in str(card_number) if d.isdigit()]
    if not digits_list:
        return False
    checksum: int = 0
    for i, digit in enumerate(reversed(digits_list)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def normalize_phone(phone_str: str) -> str:
    digits: str = str(re.sub(r"\D", "", phone_str))
    if len(digits) == 11 and digits.startswith("8"):
        return "7" + digits.replace("8", "", 1)
    if len(digits) == 10:
        return "7" + digits
    return digits


def normalize_gov_id(id_str: str) -> str:
    return str(re.sub(r"\D", "", id_str))


def is_valid_inn(inn: str) -> bool:
    digits: str = normalize_gov_id(inn)

    if len(digits) == 10:
        coeffs = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        checksum = sum(int(d) * c for d, c in zip(digits[:9], coeffs)) % 11 % 10
        return checksum == int(digits[9])

    if len(digits) == 12:
        coeffs_11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        coeffs_12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        checksum_11 = sum(int(d) * c for d, c in zip(digits[:10], coeffs_11)) % 11 % 10
        checksum_12 = sum(int(d) * c for d, c in zip(digits[:11], coeffs_12)) % 11 % 10
        return checksum_11 == int(digits[10]) and checksum_12 == int(digits[11])

    return False


def is_valid_snils(snils: str) -> bool:
    digits: str = normalize_gov_id(snils)
    if len(digits) != 11:
        return False

    total = sum(int(digit) * weight for digit, weight in zip(digits[:9], range(9, 0, -1)))
    if total < 100:
        checksum = total
    elif total in (100, 101):
        checksum = 0
    else:
        checksum = total % 101
        if checksum == 100:
            checksum = 0

    return checksum == int(digits[9:])


def clean_ocr_text_for_ner(raw_text: str) -> str:
    """Очищает мусорный текст после OCR и готовит его для Наташи."""
    text = re.sub(r"[~|`^_\\]", " ", raw_text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    normalized_words = [
        word.capitalize() if word.isupper() else word
        for word in words
    ]
    return " ".join(normalized_words)


class PIIAnalyzer:
    def __init__(self) -> None:
        self.segmenter = Segmenter() if Segmenter is not None else None
        self.morph_vocab = MorphVocab() if MorphVocab is not None else None
        self.emb = NewsEmbedding() if NewsEmbedding is not None else None
        self.ner_tagger = NewsNERTagger(self.emb) if NewsNERTagger is not None and self.emb is not None else None

        self.patterns: Dict[str, str] = {
            "passport_rf": r"\b\d{2}\s?\d{2}\s?\d{6}\b|\b\d{4}\s?\d{6}\b",
            "snils": r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}\b",
            "inn": r"\b\d{10}\b|\b\d{12}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b",
            "phone": r"\b(?:\+7|8)[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b",
            "bank_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        }
        self.passport_context_pattern = re.compile(
            r"(паспорт|серия|номер|код подразделения)",
            flags=re.IGNORECASE,
        )
        self.passport_document_pattern = re.compile(
            r"(паспорт|код подразделения|выдан|дата рождения|место рождения)",
            flags=re.IGNORECASE,
        )

    def _iter_passport_matches(self, text: str) -> Iterable[str]:
        document_looks_like_passport = bool(self.passport_document_pattern.search(text))
        for match in re.finditer(self.patterns["passport_rf"], text):
            candidate = match.group(0)
            context_start = max(0, match.start() - 160)
            context_end = min(len(text), match.end() + 160)
            context = text[context_start:context_end]
            if self.passport_context_pattern.search(context) or document_looks_like_passport:
                yield candidate

    def analyze_text(self, text: str) -> Dict[str, Any]:
        findings: Dict[str, Set[str]] = {
            "fio": set(),
            "email": set(),
            "phone": set(),
            "passport_rf": set(),
            "snils": set(),
            "inn": set(),
            "bank_card": set(),
        }

        for match in re.findall(self.patterns["email"], text):
            email_set: Optional[Set[str]] = findings.get("email")
            if email_set is not None:
                email_set.add(match.lower())

        for match in self._iter_passport_matches(text):
            passport_set: Optional[Set[str]] = findings.get("passport_rf")
            if passport_set is not None:
                passport_set.add(normalize_gov_id(match))

        for match in re.findall(self.patterns["snils"], text):
            snils_set: Optional[Set[str]] = findings.get("snils")
            if snils_set is not None and is_valid_snils(match):
                snils_set.add(normalize_gov_id(match))

        for match in re.findall(self.patterns["inn"], text):
            inn_set: Optional[Set[str]] = findings.get("inn")
            if inn_set is not None and is_valid_inn(match):
                inn_set.add(normalize_gov_id(match))

        for match in re.findall(self.patterns["bank_card"], text):
            card_set: Optional[Set[str]] = findings.get("bank_card")
            if card_set is not None and is_luhn_valid(match):
                card_set.add(normalize_gov_id(match))

        phone_matches: List[str] = re.findall(self.patterns["phone"], text)
        for phone_match in phone_matches:
            phone_set: Optional[Set[str]] = findings.get("phone")
            if phone_set is not None:
                phone_set.add(normalize_phone(phone_match))

        if all(component is not None for component in (self.segmenter, self.morph_vocab, self.ner_tagger, Doc)):
            clean_text_for_natasha = clean_ocr_text_for_ner(text)
            doc = Doc(clean_text_for_natasha)
            doc.segment(self.segmenter)
            doc.tag_ner(self.ner_tagger)

            for span in doc.spans:
                if span.type == "PER":
                    span.normalize(self.morph_vocab)
                    if span.normal and " " in span.normal:
                        fio_set: Optional[Set[str]] = findings.get("fio")
                        if fio_set is not None:
                            fio_set.add(span.normal)

        serialized_findings: Dict[str, List[str]] = {
            key: sorted(values) for key, values in findings.items()
        }

        stats: Dict[str, int] = {
            "common_pii": len(findings["fio"]) + len(findings["email"]) + len(findings["phone"]),
            "gov_ids": len(findings["passport_rf"]) + len(findings["snils"]) + len(findings["inn"]),
            "payment_info": len(findings["bank_card"]),
            "special_pii": 0,
        }

        return {
            "stats": stats,
            "raw_data": serialized_findings,
        }
