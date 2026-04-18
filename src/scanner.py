import logging
import re

from abc import ABC, abstractmethod
from dataclasses import dataclass
from natasha import Segmenter, MorphVocab, NewsNERTagger, NewsEmbedding, Doc

logger = logging.getLogger(__name__)

PROTECTION_LEVELS = {
    0:          "none",
    (1, 2):     "low",
    (3, 4):     "medium",
    (5, 6):     "high",
}

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
    """Сформировать рекомендации на основе найденных категорий."""
    recommendations = []

    critical = {"passport", "inn", "snils", "card", "driver_license"}
    biometric = {"photo", "audio", "video"}

    found = set(categories)

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

class INNDetector(RegexDetector):
    category = "inn"
    pattern  = re.compile(r"\b\d{10}(?:\d{2})?\b")


class PhoneDetector(RegexDetector):
    category = "phone"
    pattern  = re.compile(
        r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
    )


class EmailDetector(RegexDetector):
    category = "email"
    pattern  = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}")


class CardDetector(RegexDetector):
    """Номера банковских карт (16 цифр с разделителями)."""
    category = "card"
    pattern  = re.compile(r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b")


class PassportDetector(RegexDetector):
    """Серия и номер паспорта РФ: 4 цифры + 6 цифр."""
    category = "passport"
    pattern  = re.compile(r"\b\d{4}\s?\d{6}\b")


class SNILSDetector(RegexDetector):
    """СНИЛС: 123-456-789 01."""
    category = "snils"
    pattern  = re.compile(r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b")


class DateOfBirthDetector(RegexDetector):
    """Дата рождения в форматах DD.MM.YYYY и YYYY-MM-DD."""
    category = "date_of_birth"
    pattern  = re.compile(
        r"\b(?:\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})\b"
    )

""" NER-детектор для имён и адресов через natasha"""
class NERDetector(BaseDetector):
    category = "full_name"  # переопределяется динамически

    def __init__(self):
        try:
            self._segmenter  = Segmenter()
            self._morph      = MorphVocab()
            self._emb        = NewsEmbedding()
            self._tagger     = NewsNERTagger(self._emb)
            self._doc_class  = Doc
            self._available  = True
        except ImportError:
            logger.warning("natasha не установлена — NER-детектор отключён")
            self._available = False

    def detect(self, text: str) -> list[Match]:
        if not self._available:
            return []

        doc = self._doc_class(text)
        doc.segment(self._segmenter)
        doc.tag_ner(self._tagger)

        matches = []
        for span in doc.spans:
            if span.type == "PER":
                matches.append(Match(category="full_name", value=span.text))
            elif span.type == "LOC":
                matches.append(Match(category="address", value=span.text))
        return matches

class Scanner:
    """
    Запускает все детекторы и возвращает scan_markup
    """

    def __init__(self, detectors: list[BaseDetector] | None = None):
        self._detectors = detectors or [
            PassportDetector(),
            INNDetector(),
            SNILSDetector(),
            PhoneDetector(),
            EmailDetector(),
            CardDetector(),
            DateOfBirthDetector(),
            NERDetector(),
        ]

    def scan(self, text: str) -> dict:
        """
        Найти PII в тексте.

        Возвращает:
        {
            "categories":       ["passport", "inn", ...],
            "matches":          [{"type": ..., "value": ..., "location": ...}],
            "protection_level": "high",
            "recommendations":  [...],
        }
        """
        if not text.strip():
            logger.warning("Scanner получил пустой текст")
            return self._empty_markup()

        all_matches: list[Match] = []

        for detector in self._detectors:
            try:
                found = detector.detect(text)
                all_matches.extend(found)
                if found:
                    logger.info(
                        "Детектор [%s]: найдено %d вхождений",
                        detector.category, len(found),
                    )
            except Exception as e:
                logger.error("Детектор [%s] упал: %s", detector.category, e)

        categories = list({m.category for m in all_matches})
        protection_level = _calc_protection_level(categories)
        recommendations  = _calc_recommendations(categories)

        logger.info(
            "Сканирование завершено: категорий=%d, вхождений=%d, уровень=%s",
            len(categories), len(all_matches), protection_level,
        )

        return {
            "categories":       categories,
            "matches":          [
                {"type": m.category, "value": m.value, "location": m.location}
                for m in all_matches
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
