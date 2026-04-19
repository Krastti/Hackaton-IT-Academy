import logging
import difflib
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pymorphy3
sys.modules["pymorphy2"] = pymorphy3
sys.modules["pymorphy2.analyzer"] = pymorphy3.analyzer

from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger, Doc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательные функции валидации и нормализации
# ---------------------------------------------------------------------------

def _is_luhn_valid(card: str) -> bool:
    """Алгоритм Луна — проверка номера банковской карты."""
    digits = [int(d) for d in card if d.isdigit()]
    if len(digits) < 13:
        return False
    odd_sum = sum(d * 2 - 9 if d * 2 > 9 else d * 2 for d in digits[-2::-2])
    even_sum = sum(digits[-1::-2])
    return (odd_sum + even_sum) % 10 == 0


def _is_snils_valid(snils: str) -> bool:
    """Проверка контрольной суммы СНИЛС."""
    digits = re.sub(r"\D", "", snils)
    if len(digits) != 11:
        return False
    num, control = digits[:9], int(digits[9:])
    if int(num) <= 1001998:
        return True
    checksum = sum(int(d) * (9 - i) for i, d in enumerate(num)) % 101
    return (0 if checksum in (100, 101) else checksum) == control


def _normalize_phone(phone: str) -> str:
    """Приводит телефон к формату 7XXXXXXXXXX."""
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 11 and digits.startswith("8"):
        return "7" + digits[1:]
    return "7" + digits if len(digits) == 10 else digits


def _heal_ocr_digits(text: str) -> str:
    """Исправляет типичные OCR-ошибки в цифрах (O→0, З→3, В→8 и т.д.)"""
    return text.translate(str.maketrans("OоОЗзBВIlS", "0003038115"))


def _deduplicate(items: set, threshold: float = 0.85) -> set:
    """Дедупликация через нечёткое сравнение строк."""
    unique = []
    for item in items:
        for i, u in enumerate(unique):
            if difflib.SequenceMatcher(None, item, u).ratio() > threshold:
                if len(item) > len(u):
                    unique[i] = item
                break
        else:
            unique.append(item)
    return set(unique)


def _clean_for_ner(text: str) -> str:
    """Очищает OCR-текст перед подачей в NER."""
    text = re.sub(r"[~|`^_\\]", " ", text)
    return " ".join(w.capitalize() for w in re.sub(r"[a-zA-Z]", "", text).split())


# ---------------------------------------------------------------------------
# Уровень защиты и рекомендации
# ---------------------------------------------------------------------------

def _calc_protection_level(categories: list[str]) -> str:
    """Определить уровень защиты по количеству найденных категорий."""
    count = len(categories)
    if count == 0:
        return "none"
    if count <= 2:
        return "low"
    if count <= 4:
        return "medium"
    if count <= 6:
        return "high"
    return "critical"

def _calc_recommendations(categories: list[str]) -> list[str]:
    found = set(categories)
    recommendations = []
    critical  = {"passport", "inn", "snils", "bank_card", "driver_license"}
    biometric = {"special_pii"}

    if found & critical:
        recommendations.append("Ограничить доступ к файлу")
        recommendations.append("Зашифровать хранилище")
    if found & biometric:
        recommendations.append("Применить законодательство о биометрических данных")
    if "email" in found or "phone" in found:
        recommendations.append("Уведомить владельца данных")
    if len(found) >= 3:
        recommendations.append("Провести аудит доступа")

    return recommendations

@dataclass
class Match:
    category: str
    value: str
    location: str | None = None

class BaseDetector(ABC):
    category: str

    @abstractmethod
    def detect(self, text: str) -> list[Match]:
        """Находит вхождения ПДн в тексте"""

class RegexDetector(BaseDetector):
    """Базовый класс для regex-детекторов."""

    pattern: re.Pattern

    def detect(self, text: str) -> list[Match]:
        matches = []
        for m in self.pattern.finditer(text):
            line_num = text[: m.start()].count("\n") + 1
            matches.append(Match(
                category=self.category,
                value=m.group(),
                location=f"строка {line_num}",
            ))
        return matches


# ---------------------------------------------------------------------------
# Regex-детекторы
# ---------------------------------------------------------------------------

class PassportRFDetector(RegexDetector):
    """Паспорт РФ: серия (4 цифры) + номер (6 цифр)."""
    category = "passport"
    pattern  = re.compile(
        r"(?i)(?:"
        r"(?:п[аa]сп[оa]рт(?:ные\s+данные)?|серия(?:\s+и\s+номер)?)[:\s#№]{0,10}"
        r"\d{2}[\s\-\.,_]{0,5}\d{2}[\s\-\.,_]{0,5}\d{6}"
        r"|"
        r"(?<!\d)(?:\d{2}[\s\-\.,_]{1,5}\d{2}[\s\-\.,_]{1,5}\d{6})(?!\d)"
        r")"
    )


class PassportIntlDetector(RegexDetector):
    """Международный паспорт / MRZ строка."""
    category = "passport"
    pattern  = re.compile(
        r"(?i)(?:"
        r"(?:P|I|A|C|V)[A-Z<]{1,2}[A-Z<]{3}[A-Z0-9<]{10,50}"
        r"|"
        r"(?:passport|document\s*no\.?)[^A-Za-z0-9А-Яа-я]{0,10}[A-Z0-9]{6,12}\b"
        r")"
    )


class INNDetector(RegexDetector):
    category = "inn"
    pattern  = re.compile(r"(?i)(?:инн)[:\s#№]{0,10}[\d]{10,12}|\b\d{12}\b")


class PhoneDetector(RegexDetector):
    category = "phone"
    pattern  = re.compile(
        r"\b(?:\+7|8)[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b"
    )


class EmailDetector(RegexDetector):
    category = "email"
    # Лимиты длины защищают от зависания на строках с тысячами точек
    pattern  = re.compile(
        r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,64}\.[A-Za-z]{2,7}"
    )


class DriverLicenseDetector(RegexDetector):
    category = "driver_license"
    pattern  = re.compile(r"\b\d{2}[\s]?[А-ЯA-Z]{2}[\s]?\d{6}\b")


class DateOfBirthDetector(RegexDetector):
    """Дата рождения в форматах DD.MM.YYYY и YYYY-MM-DD."""
    category = "date_of_birth"
    pattern  = re.compile(
        r"\b(?:\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})\b"
    )


class BankCardDetector(BaseDetector):
    """Банковская карта с валидацией по алгоритму Луна."""
    category = "bank_card"
    _pattern = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

    def detect(self, text: str) -> list[Match]:
        matches = []
        for m in self._pattern.finditer(text):
            if _is_luhn_valid(m.group()):
                line_num = text[: m.start()].count("\n") + 1
                matches.append(Match(
                    category=self.category,
                    value=re.sub(r"\D", "", m.group()),
                    location=f"строка {line_num}",
                ))
        return matches


class SNILSDetector(BaseDetector):
    """СНИЛС с валидацией контрольной суммы."""
    category = "snils"
    _pattern = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]\d{2}\b")

    def detect(self, text: str) -> list[Match]:
        matches = []
        for m in self._pattern.finditer(text):
            if _is_snils_valid(m.group()):
                line_num = text[: m.start()].count("\n") + 1
                matches.append(Match(
                    category=self.category,
                    value=re.sub(r"\D", "", m.group()),
                    location=f"строка {line_num}",
                ))
        return matches


class AddressDetector(BaseDetector):
    """Адреса по ключевым словам (ул., д., г. и т.д.)"""
    category = "address"
    _pattern = re.compile(
        r"(?:г\.|город|ул\.|улица|пр-?т\.?|проспект|пер\.|переулок|"
        r"б-?р\.?|бульвар|пл\.|площадь|шоссе|набережная)"
        r"[\s\w\-«»\"'.]{1,100}?(?:д\.|дом)\s*\d{1,5}\w{0,5}",
        re.IGNORECASE,
    )

    def detect(self, text: str) -> list[Match]:
        matches = []
        for m in self._pattern.finditer(text):
            line_num = text[: m.start()].count("\n") + 1
            matches.append(Match(
                category=self.category,
                value=re.sub(r"\s+", " ", m.group()).strip(),
                location=f"строка {line_num}",
            ))
        return matches


class SpecialPIIDetector(BaseDetector):
    """
    Специальные категории ПДн: медицина, биометрия, религия, национальность.
    \w{0,30} вместо \w+ защищает от зависания на длинных строках.
    """
    category = "special_pii"
    _pattern = re.compile("|".join([
        r"\bдиагноз", r"\bдиабет", r"\bонкол\w{0,30}", r"\bвич\b", r"\bспид\b",
        r"\bинвалид\w{0,30}", r"\bгруппа\s{1,5}крови", r"\bмкб[-\s]?\d{1,5}",
        r"\bрецепт\b", r"\bлечени\w{0,30}", r"\bгоспитализац\w{0,30}",
        r"\bбиометр\w{0,30}", r"\bдактилоскоп\w{0,30}", r"\bраспознавание\s{1,5}лиц",
        r"\bface[\s_-]{0,5}id\b", r"\bfingerprint\b",
        r"\bрелиги\w{0,30}", r"\bвероисповедан\w{0,30}",
        r"\bнациональност\w{0,30}", r"\bрасов\w{0,30}", r"\bэтническ\w{0,30}",
    ]), re.IGNORECASE)

    def detect(self, text: str) -> list[Match]:
        matches = []
        for m in self._pattern.finditer(text):
            line_num = text[: m.start()].count("\n") + 1
            matches.append(Match(
                category=self.category,
                value=m.group().strip().lower(),
                location=f"строка {line_num}",
            ))
        return matches


# ---------------------------------------------------------------------------
# NER-детектор
# ---------------------------------------------------------------------------

class NERDetector(BaseDetector):
    category = "full_name"  # переопределяется динамически

    def __init__(self):
        try:
            self._segmenter = Segmenter()
            self._morph     = MorphVocab()
            self._emb       = NewsEmbedding()
            self._tagger    = NewsNERTagger(self._emb)
            self._available = True
        except Exception as e:
            logger.warning("NERDetector недоступен: %s", e)
            self._available = False

    def detect(self, text: str) -> list[Match]:
        if not self._available:
            return []

        safe_text = _clean_for_ner(text[:5000])
        doc = Doc(safe_text)
        doc.segment(self._segmenter)
        doc.tag_ner(self._tagger)

        fio_candidates: dict[str, str] = {}
        matches = []

        for span in doc.spans:
            if span.type == "PER":
                span.normalize(self._morph)
                words = (span.normal or span.text).split()[:3]
                if len(words) >= 2 and all(len(w) >= 4 for w in words):
                    name = " ".join(words)
                    key  = words[0].lower()
                    if len(name) > len(fio_candidates.get(key, "")):
                        fio_candidates[key] = name
            elif span.type == "LOC":
                matches.append(Match(category="address", value=span.text))

        for name in fio_candidates.values():
            matches.append(Match(category="full_name", value=name))

        return matches

class Scanner:
    """
    Запускает все детекторы и возвращает scan_markup
    """

    def __init__(self, detectors: list[BaseDetector] | None = None):
        self._detectors = detectors or [
            PassportRFDetector(),
            PassportIntlDetector(),
            INNDetector(),
            SNILSDetector(),
            PhoneDetector(),
            EmailDetector(),
            BankCardDetector(),
            DriverLicenseDetector(),
            DateOfBirthDetector(),
            AddressDetector(),
            SpecialPIIDetector(),
            NERDetector(),
        ]

    def scan(self, text: str) -> dict:
        if not text.strip():
            logger.warning("Scanner получил пустой текст")
            return self._empty_markup()

        # Лимит текста — защита от зависания
        if len(text) > 500_000:
            logger.warning("Текст обрезан до 500к символов")
            text = text[:500_000]

        # Числовые детекторы работают с OCR-исправленным текстом
        text_healed = _heal_ocr_digits(text)
        numeric_detectors = {
            "passport", "inn", "snils", "phone", "bank_card",
            "driver_license", "date_of_birth", "address",
        }

        all_matches: list[Match] = []

        for detector in self._detectors:
            try:
                src = text_healed if detector.category in numeric_detectors else text
                found = detector.detect(src)
                all_matches.extend(found)
                if found:
                    logger.info(
                        "Детектор [%s]: найдено %d вхождений",
                        detector.category, len(found),
                    )
            except Exception as e:
                logger.error("Детектор [%s] упал: %s", detector.category, e)

        # Дедупликация для полей с высоким риском дублей
        dedup_categories = {"passport", "snils", "inn", "driver_license", "full_name"}
        deduped: dict[str, set] = {}
        clean_matches: list[Match] = []

        for m in all_matches:
            if m.category in dedup_categories:
                deduped.setdefault(m.category, set()).add(m.value)
            else:
                clean_matches.append(m)

        for cat, values in deduped.items():
            for v in _deduplicate(values):
                clean_matches.append(Match(category=cat, value=v))

        categories = list({m.category for m in clean_matches})
        protection_level = _calc_protection_level(categories)
        recommendations  = _calc_recommendations(categories)

        logger.info(
            "Сканирование завершено: категорий=%d, вхождений=%d, уровень=%s",
            len(categories), len(clean_matches), protection_level,
        )

        return {
            "categories":       categories,
            "matches":          [
                {"type": m.category, "value": m.value, "location": m.location}
                for m in clean_matches
            ],
            "protection_level": protection_level,
            "recommendations":  recommendations,
        }

    @staticmethod
    def _empty_markup() -> dict:
        return {
            "categories":       [],
            "matches":          [],
            "protection_level": "none",
            "recommendations":  [],
        }
