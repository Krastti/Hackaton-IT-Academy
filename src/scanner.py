import re
from typing import Dict, Set, List, Any, Optional
import sys
import pymorphy3

sys.modules['pymorphy2'] = pymorphy3
sys.modules['pymorphy2.analyzer'] = pymorphy3.analyzer

from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger, Doc  # type: ignore


# ---------------------------------------------------------------------------
# Вспомогательные валидаторы
import difflib
# ---------------------------------------------------------------------------

def is_luhn_valid(card_number: str) -> bool:
    """Алгоритм Луна для банковских карт."""
    digits_list: List[int] = [int(d) for d in str(card_number) if d.isdigit()]
    if len(digits_list) < 13:
        return False
    checksum: int = 0
    for i, digit in enumerate(reversed(digits_list)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def is_snils_valid(snils_digits: str) -> bool:
    """Проверка контрольной суммы СНИЛС (11 цифр без разделителей)."""
    if len(snils_digits) != 11 or not snils_digits.isdigit():
        return False
    number = snils_digits[:9]
    control = int(snils_digits[9:11])
    # Номера до 001-001-998 — специальные, контрольная сумма не проверяется
    if int(number) <= 1001998:
        return True
    checksum: int = 0
    for i, d in enumerate(number):
        checksum += int(d) * (9 - i)
    if checksum > 101:
        checksum %= 101
    if checksum in (100, 101):
        checksum = 0
    return checksum == control


def normalize_phone(phone_str: str) -> str:
    digits: str = re.sub(r'\D', '', phone_str)
    if len(digits) == 11 and digits.startswith('8'):
        return '7' + digits[1:]
    elif len(digits) == 10:
        return '7' + digits
    return digits


def normalize_gov_id(id_str: str) -> str:
    return re.sub(r'\D', '', id_str)


# ---------------------------------------------------------------------------
# Детектор специальных категорий ПДн и биометрии (152-ФЗ ст. 10 и ст. 11)
# ---------------------------------------------------------------------------

_SPECIAL_KW: List[str] = [
    # Состояние здоровья / медицина
    r'диагноз', r'диабет', r'онкол\w+', r'\bвич\b', r'\bспид\b',
    r'инвалид\w+', r'группа\s+крови', r'анализ\s+крови',
    r'мкб[-\s]?\d+', r'история\s+болезни', r'медицинск\w+\s+карт\w+',
    r'рецепт\b', r'лечени\w+', r'госпитализац\w+', r'хирург\w+',
    r'операци\w+', r'психиатр\w+', r'нарколог\w+', r'иммунодефицит',
    # Биометрия
    r'отпеч\w+\s+пальц', r'радужн\w+\s+оболочк', r'сетчатк\w+',
    r'голосовой\s+образец', r'биометр\w+', r'дактилоскоп\w+',
    r'распознавание\s+лиц', r'face[\s_-]*id', r'fingerprint',
    # Религия и политические убеждения
    r'религи\w+', r'вероисповедан\w+', r'конфесси\w+', r'православ\w+',
    r'мусульман\w+', r'католи\w+', r'атеист\w+', r'буддист\w+',
    r'политическ\w+\s+(?:взгляд|убежден|позиц)\w+',
    # Национальность и раса
    r'национальност\w+', r'расов\w+', r'этническ\w+',
]

SPECIAL_PII_RE: re.Pattern = re.compile(
    '|'.join(_SPECIAL_KW), re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Паттерн адресов
# ---------------------------------------------------------------------------

ADDRESS_RE: re.Pattern = re.compile(
    r'(?:'
    r'г\.|город|ул\.|улица|пр-?т\.?|проспект|пер\.|переулок'
    r'|б-?р\.?|бульвар|пл\.|площадь|шоссе|набережная'
    r')'
    r'[\s\w\-«»"\'.]+?'
    r'(?:д\.|дом)\s*\d+\w*',
    re.IGNORECASE
)


def clean_ocr_text_for_ner(raw_text: str) -> str:
    """Очищает текст ТОЛЬКО для NER (Natasha). НЕ применять к тексту перед regex."""
    text = re.sub(r'[~|`^_\\]', ' ', raw_text)
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    # Title Case для каждого слова — Natasha ищет имена с заглавной
    normalized_words = [
        word.capitalize()  # работает и на ALL_CAPS, и на строчных
        for word in words
    ]
    return " ".join(normalized_words)


# ---------------------------------------------------------------------------
# Основной анализатор
# ---------------------------------------------------------------------------

class PIIAnalyzer:
    def __init__(self) -> None:
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.emb)

        self.patterns: Dict[str, str] = {
            # Паспорт РФ: серия (4 цифры) + номер (6 цифр)
            # Варианты из OCR:
            #  1) С ключевым словом «паспорт/серия» + цифры
            #  2) Обычный формат «XX XX XXXXXX» / «XX-XX-XXXXXX» / «XXXX XXXXXX»
            #  3) MRZ (ICAO TD3): зона машинного чтения — цифры+RUS+цифры, 
            #     напр.: 4528336887RU80803119<<<<<< или 887RU80803119
            'passport_rf': (
                r'(?i)(?:паспорт(?:ные\s+данные)?|серия(?:\s+и\s+номер)?)[:\s#№]*'
                r'\d{2}\s?\d{2}[\s\-]?\d{6}'
                r'|\b\d{2}[\s\-]?\d{2}[\s\-]\d{6}\b'        # XX XX XXXXXX / XX-XX-XXXXXX
                r'|\b\d{4}[\s\-]\d{6}\b'                    # XXXX XXXXXX
                r'|\b\d{9,10}(?:RU|RUS)\d{7,9}[<A-Z0-9]*'  # MRZ: NNNNNNNNNRUSDDDDDDDD<<
                r'|\d{3,4}(?:RU|RUS)\d{7,9}[<A-Z0-9]'      # MRZ обрезанный: 887RU80803119
            ),
            # СНИЛС: 11 цифр с возможными разделителями — XXX-XXX-XXX XX
            'snils': r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]\d{2}\b',
            # ИНН: 10 цифр — только с контекстом; 12 цифр — без контекста
            'inn': (
                r'(?i)(?:инн)[:\s#№]*[\d]{10,12}'
                r'|\b\d{12}\b'
            ),
            'email': r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,7}\b',
            'phone': r'\b(?:\+7|8)[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b',
            'bank_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            # Водительское удостоверение РФ: 2 цифры + 2 буквы (рус/лат) + 6 цифр
            'driver_license': r'\b\d{2}[\s]?[А-ЯA-Z]{2}[\s]?\d{6}\b',
            # БИК: 9 цифр, начинается с 04, с контекстом
            'bik': r'(?i)(?:бик|BIK)[:\s]*\b04\d{7}\b',
            # CVV/CVC: 3-4 цифры с контекстом
            'cvv': r'(?i)(?:cvv2?|cvc2?|csc)[:\s]*\b\d{3,4}\b',
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        findings: Dict[str, Set[str]] = {
            'fio': set(),
            'email': set(),
            'phone': set(),
            'passport_rf': set(),
            'snils': set(),
            'inn': set(),
            'bank_card': set(),
            'driver_license': set(),
            'bik': set(),
            'cvv': set(),
            'address': set(),
        }

        # --- 1. Regex-поиск на оригинальном тексте (латиница не удаляется) ---
        for key in ['email', 'bank_card', 'driver_license']:
            for match in re.findall(self.patterns[key], text):
                t = findings.get(key)
                if t is None:
                    continue
                if key == 'bank_card':
                    if is_luhn_valid(match):
                        t.add(normalize_gov_id(match))
                else:
                    t.add(match.strip())

        for key in ['passport_rf', 'snils', 'inn']:
            for match in re.findall(self.patterns[key], text):
                t = findings.get(key)
                if t is None:
                    continue
                normalized = normalize_gov_id(match)
                if key == 'snils':
                    if is_snils_valid(normalized):
                        t.add(normalized)
                else:
                    t.add(normalized)

        for p in re.findall(self.patterns['phone'], text):
            s = findings.get('phone')
            if s is not None:
                s.add(normalize_phone(p))

        for key in ['bik', 'cvv']:
            for match in re.findall(self.patterns[key], text):
                t = findings.get(key)
                if t is not None:
                    digits_only = re.sub(r'\D', '', match)
                    if digits_only:
                        t.add(digits_only)

        for match in ADDRESS_RE.findall(text):
            addr_set = findings.get('address')
            if addr_set is not None:
                addr_set.add(re.sub(r'\s+', ' ', match).strip())

        # --- 2. Специальные категории ПДн (ключевые слова) ---
        special_matches: List[str] = SPECIAL_PII_RE.findall(text)
        special_count: int = len(special_matches)

        # Словарь для дедупликации ФИО по первым символам фамилии
        fio_by_surname: Dict[str, str] = {}

        clean_text_for_ner = clean_ocr_text_for_ner(text)
        doc = Doc(clean_text_for_ner)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)

        # Дедупликация ФИО по фамилии: одно и то же имя может быть найдено
        # в разных кадрах видео в разных падежах → разные строки в set.
        # Группируем по первому слову (фамилии), оставляем самую длинную форму.
        for span in doc.spans:
            if span.type == 'PER':
                span.normalize(self.morph_vocab)
                if span.normal:
                    words = span.normal.split()
                    # Берём максимум первые 3 слова (Фамилия Имя Отчество)
                    # Natasha иногда захватывает лишние слова после имени
                    fio_words = words[:3]
                    # Каждое слово ≥ 4 символа (отсекает OCR-мусор «Муж», «Гор»)
                    if len(fio_words) >= 2 and all(len(w) >= 4 for w in fio_words):
                        name_str = ' '.join(fio_words)
                        surname_key = fio_words[0].lower()
                        existing = fio_by_surname.get(surname_key, '')
                        if len(name_str) > len(existing):
                            fio_by_surname[surname_key] = name_str

        fio_set_final: Optional[Set[str]] = findings.get('fio')
        if fio_set_final is not None:
            final_fio_list = []
            for name in fio_by_surname.values():
                is_similar = False
                for i, ext_name in enumerate(final_fio_list):
                    if difflib.SequenceMatcher(None, name.lower(), ext_name.lower()).ratio() > 0.6:
                        if len(name) > len(ext_name):
                            final_fio_list[i] = name
                        is_similar = True
                        break
                if not is_similar:
                    final_fio_list.append(name)
            fio_set_final.update(final_fio_list)

        serialized_findings: Dict[str, List[str]] = {
            k: list(v) for k, v in findings.items()
        }

        stats: Dict[str, int] = {
            'common_pii': (
                len(findings['fio']) + len(findings['email']) +
                len(findings['phone']) + len(findings['address'])
            ),
            'gov_ids': (
                len(findings['passport_rf']) + len(findings['snils']) +
                len(findings['inn']) + len(findings['driver_license'])
            ),
            'payment_info': (
                len(findings['bank_card']) + len(findings['bik']) + len(findings['cvv'])
            ),
            'special_pii': special_count,
        }

        return {
            'stats': stats,
            'raw_data': serialized_findings,
        }