import re
import sys
import difflib
from typing import Dict, List, Any, Optional

import pymorphy3

sys.modules['pymorphy2'] = pymorphy3
sys.modules['pymorphy2.analyzer'] = pymorphy3.analyzer

from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger, Doc


class Scanner:
    def __init__(self):
        self.analyzer = PIIAnalyzer()

    def scan(self, text: str) -> dict:
        """
        Метод вызывается из app.py.
        Преобразует результаты PIIAnalyzer в формат, ожидаемый batch.to_report()
        """
        if not text:
            return self._empty_result()

        analysis = self.analyzer.analyze_text(text)
        raw_data = analysis.get("raw_data", {})
        stats = analysis.get("stats", {})

        # Определяем найденные категории (где список не пуст)
        categories = [k for k, v in raw_data.items() if v]

        # Собираем все найденные совпадения в один список для подсчета total_matches
        all_matches = []
        for val in raw_data.values():
            all_matches.extend(val)

        # Логика определения уровня защиты
        protection_level = "Low"
        recommendations = ["Хранение в открытом виде допустимо."]

        if stats.get("special_pii", 0) > 0 or stats.get("gov_ids", 0) > 0:
            protection_level = "High"
            recommendations = ["Критично: Требуется шифрование и строгий аудит доступа.", "Анонимизировать данные."]
        elif stats.get("common_pii", 0) > 0 or stats.get("payment_info", 0) > 0:
            protection_level = "Medium"
            recommendations = ["Рекомендуется ограничить доступ.", "Проверить целесообразность хранения."]

        return {
            "categories": categories,
            "matches": all_matches,
            "protection_level": protection_level,
            "recommendations": recommendations,
            "raw_stats": stats  # для отладки
        }

    def _empty_result(self) -> dict:
        return {
            "categories": [],
            "matches": [],
            "protection_level": "None",
            "recommendations": ["Файл пуст или текст не извлечен."]
        }


def is_luhn_valid(card: str) -> bool:
    digits = [int(d) for d in str(card) if d.isdigit()]
    if len(digits) < 13:
        return False

    odd_sum = sum(d * 2 - 9 if d * 2 > 9 else d * 2 for d in digits[-2::-2])
    even_sum = sum(digits[-1::-2])
    return (odd_sum + even_sum) % 10 == 0


def is_snils_valid(snils: str) -> bool:
    if len(snils) != 11 or not snils.isdigit():
        return False

    num, control = snils[:9], int(snils[9:])
    if int(num) <= 1001998:
        return True

    checksum = sum(int(d) * (9 - i) for i, d in enumerate(num)) % 101
    return (0 if checksum in (100, 101) else checksum) == control


def normalize_phone(phone: str) -> str:
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 11 and digits.startswith('8'):
        return '7' + digits[1:]
    return '7' + digits if len(digits) == 10 else digits


def normalize_gov_id(gov_id: str) -> str:
    return re.sub(r'\D', '', gov_id)


def clean_ocr_text_for_ner(raw_text: str) -> str:
    text = re.sub(r'[~|`^_\\]', ' ', raw_text)
    return " ".join(word.capitalize() for word in re.sub(r'[a-zA-Z]', '', text).split())


# ИСПРАВЛЕНИЕ 1: Жесткие ограничения на длину слов (\w{0,30} вместо \w+)
_SPECIAL_KW = [
    r'\bдиагноз', r'\bдиабет', r'\bонкол\w{0,30}', r'\bвич\b', r'\bспид\b',
    r'\bинвалид\w{0,30}', r'\bгруппа\s{1,5}крови', r'\bанализ\s{1,5}крови',
    r'\bмкб[-\s]?\d{1,5}', r'\бистория\s{1,5}болезни', r'\bмедицинск\w{0,30}\s{1,5}карт\w{0,30}',
    r'\bрецепт\b', r'\bлечени\w{0,30}', r'\bгоспитализац\w{0,30}', r'\bхирург\w{0,30}',
    r'\bопераци\w{0,30}', r'\бпсихиатр\w{0,30}', r'\bнарколог\w{0,30}', r'\bиммунодефицит',
    r'\bотпеч\w{0,30}\s{1,5}пальц', r'\bрадужн\w{0,30}\s{1,5}оболочк', r'\bсетчатк\w{0,30}',
    r'\bголосовой\s{1,5}образец', r'\ббиометр\w{0,30}', r'\bдактилоскоп\w{0,30}',
    r'\bраспознавание\s{1,5}лиц', r'\bface[\s_-]{0,5}id\b', r'\bfingerprint\b',
    r'\bрелиги\w{0,30}', r'\bвероисповедан\w{0,30}', r'\bконфесси\w{0,30}', r'\bправослав\w{0,30}',
    r'\bмусульман\w{0,30}', r'\bкатоли\w{0,30}', r'\bатеист\w{0,30}', r'\bбуддист\w{0,30}',
    r'\bполитическ\w{0,30}\s{1,5}(?:взгляд|убежден|позиц)\w{0,30}',
    r'\bнациональност\w{0,30}', r'\bрасов\w{0,30}', r'\bэтническ\w{0,30}'
]

SPECIAL_PII_RE = re.compile('|'.join(_SPECIAL_KW), re.IGNORECASE)

# ИСПРАВЛЕНИЕ 2: Защита от поиска адресов в мегабайтах мусора
ADDRESS_RE = re.compile(
    r'(?:г\.|город|ул\.|улица|пр-?т\.?|проспект|пер\.|переулок|'
    r'б-?р\.?|бульвар|пл\.|площадь|шоссе|набережная)'
    r'[\s\w\-«»"\'.]{1,100}?(?:д\.|дом)\s*\d{1,5}\w{0,5}',
    re.IGNORECASE
)


class PIIAnalyzer:
    PATTERNS = {
        # ИСПРАВЛЕНИЕ 3: Защита от бесконечного перебора пробелов
        'passport_rf': (
            r'(?i)(?:'
            r'(?:п[аaоo0]сп[оoаa0]рт(?:ные\s+данные)?|серия(?:\s+и\s+номер)?)[:\s#№]{0,10}'
            r'\d{2}[\s\-\.,_]{0,5}\d{2}[\s\-\.,_]{0,5}\d{6}'
            r'|'
            r'(?<!\d)(?:\d{2}[\s\-\.,_]{1,5}\d{2}[\s\-\.,_]{1,5}\d{6})(?!\d)'
            r')'
        ),
        'passport_intl': (
            r'(?i)(?:'
            r'(?:P|I|A|C|V)[A-Z<]{1,2}[A-Z<]{3}[A-Z0-9<]{10,50}'
            r'|'
            r'(?:passport|document\s*no\.?)[^A-Za-z0-9А-Яа-я]{0,10}[A-Z0-9]{6,12}\b'
            r')'
        ),
        'snils': r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]\d{2}\b',
        'inn': r'(?i)(?:инн)[:\s#№]{0,10}[\d]{10,12}|\b\d{12}\b',
        # ИСПРАВЛЕНИЕ 4: Лимиты на email, чтобы не зависать на строках из 1000 точек
        'email': r'[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,64}\.[A-Za-z]{2,7}',
        'phone': r'\b(?:\+7|8)[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b',
        'bank_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'driver_license': r'\b\d{2}[\s]?[А-ЯA-Z]{2}[\s]?\d{6}\b',
        'bik': r'(?i)(?:бик|BIK)[:\s]{0,10}\b04\d{7}\b',
        'cvv': r'(?i)(?:cvv2?|cvc2?|csc)[:\s]{0,10}\b\d{3,4}\b',
    }

    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.emb)

    def _deduplicate_ids(self, items: set, threshold: float = 0.85) -> set:
        unique_items = []
        for item in items:
            is_duplicate = False
            for u_item in unique_items:
                if difflib.SequenceMatcher(None, item, u_item).ratio() > threshold:
                    if len(item) > len(u_item):
                        unique_items[unique_items.index(u_item)] = item
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_items.append(item)

        mrz_items = [it for it in unique_items if re.match(r'(?i)^[PIACV][A-Z]', it) and len(it) > 15]
        visual_items = [it for it in unique_items if it not in mrz_items]

        if mrz_items and visual_items:
            return set(visual_items)

        return set(unique_items)

    def _extract_names_from_mrz(self, text: str) -> set:
        names = set()
        # Обрезаем входной текст, чтобы не пытаться удалять пробелы из 50 МБ
        safe_text = text[:10000].upper()
        no_space_text = re.sub(r'[\s\n]+', '', safe_text)

        mrz_match = re.search(
            r'(?:P|I|A|C|V)[A-Z<KCE«\(]{1,2}[A-Z]{3}([A-Z0-9]+)[<KCE«\(]{2}([A-Z0-9<KCE«\(]+)',
            no_space_text
        )

        if mrz_match:
            surname_raw = mrz_match.group(1)
            given_raw = mrz_match.group(2)

            clean_surname = re.sub(r'[<KCE«\(]+', ' ', surname_raw).strip()
            given_split = re.split(r'[<KCE«\(]{3,}', given_raw)
            clean_given = re.sub(r'[<KCE«\(]+', ' ', given_split[0]).strip()

            if clean_surname and clean_given:
                tr_map = str.maketrans('8301', 'YZOI')
                clean_surname = clean_surname.translate(tr_map)
                clean_given = clean_given.translate(tr_map)

                names.add(f"{clean_surname} {clean_given}".title())

        return names

    def _extract_foreign_pii_by_anchors(self, text: str) -> Dict[str, set]:
        anchored_data = {'fio': set(), 'passport': set()}

        # ИСПРАВЛЕНИЕ 5: Ограничиваем якоря, чтобы не зависать
        surnames = re.findall(r'(?i)\b(?:surname|last name|nom|фамилия)[\s:;\.\|]{1,20}([A-ZА-Я]{2,}(?:[- ][A-ZА-Я]{2,})?)', text)
        names = re.findall(r'(?i)\b(?:given names?|first name|name|pr[eé]noms?|имя)[\s:;\.\|]{1,20}([A-ZА-Я]{2,}(?:[\s-][A-ZА-Я]{2,})?)', text)

        if surnames and names:
            anchored_data['fio'].add(f"{surnames[0].strip()} {names[0].strip()}".title())
        elif surnames:
            anchored_data['fio'].add(surnames[0].strip().title())
        elif names:
            anchored_data['fio'].add(names[0].strip().title())

        doc_nos = re.findall(
            r'(?i)\b(?:document no|passport no|id no|identity no|document number|номер документа)[\s:;\.\|№#]{1,20}([A-Z0-9]{6,14})\b',
            text)
        for doc in doc_nos:
            cleaned = re.sub(r'[^A-Z0-9]', '', doc.upper())
            if len(cleaned) >= 6:
                anchored_data['passport'].add(cleaned)

        return anchored_data

    def _heal_ocr_digits(self, text: str) -> str:
        replacements = {
            'O': '0', 'о': '0', 'О': '0',
            'З': '3', 'з': '3',
            'B': '8', 'В': '8',
            'I': '1', 'l': '1',
            'S': '5'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _process_match(self, key: str, match: str) -> Optional[str]:
        if key == 'bank_card':
            return normalize_gov_id(match) if is_luhn_valid(match) else None
        if key == 'snils':
            norm = normalize_gov_id(match)
            return norm if is_snils_valid(norm) else None

        if key == 'passport':
            cleaned = re.sub(r'(?i)(паспортные данные|паспорт|серия и номер|серия|passport|document no\.?)', '', match)
            cleaned = re.sub(r'[^A-Za-zА-Яа-я0-9]', '', cleaned)
            return cleaned if len(cleaned) >= 6 else None

        if key == 'inn':
            return normalize_gov_id(match)
        if key == 'phone':
            return normalize_phone(match)
        if key in ('bik', 'cvv'):
            digits = re.sub(r'\D', '', match)
            return digits if digits else None
        if key in ('email', 'driver_license'):
            return match.strip()
        return None

    def _extract_fios(self, text: str) -> List[str]:
        # Нейросеть не выдержит больше 50к символов
        safe_text = text[:5000]
        doc = Doc(clean_ocr_text_for_ner(safe_text))
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)

        fio_candidates = {}
        for span in doc.spans:
            if span.type != 'PER':
                continue

            span.normalize(self.morph_vocab)
            if not span.normal:
                continue

            words = span.normal.split()[:3]
            if len(words) >= 2 and all(len(w) >= 4 for w in words):
                name = ' '.join(words)
                k = words[0].lower()
                if len(name) > len(fio_candidates.get(k, '')):
                    fio_candidates[k] = name

        final_fios = []
        for name in fio_candidates.values():
            for i, ext in enumerate(final_fios):
                if difflib.SequenceMatcher(None, name.lower(), ext.lower()).ratio() > 0.6:
                    if len(name) > len(ext):
                        final_fios[i] = name
                    break
            else:
                final_fios.append(name)

        return final_fios

    def analyze_text(self, text: str) -> Dict[str, Any]:
        if len(text) > 500000:
            text = text[:500000]

        actual_keys = set(k if not k.startswith('passport') else 'passport' for k in self.PATTERNS)

        # 1. ДОБАВЛЕНО 'special_pii' В МАССИВ FINDINGS
        findings = {k: set() for k in actual_keys | {'fio', 'address', 'special_pii'}}

        text_for_digits = self._heal_ocr_digits(text)

        for key, pattern in self.PATTERNS.items():
            target_text = text if key in ('email', 'driver_license', 'passport_intl') else text_for_digits
            for match in re.findall(pattern, target_text):
                save_key = 'passport' if key.startswith('passport') else key
                processed = self._process_match(save_key, match)
                if processed:
                    findings[save_key].add(processed)

        for match in ADDRESS_RE.findall(text):
            findings['address'].add(re.sub(r'\s+', ' ', match).strip())

        # 2. СОБИРАЕМ САМИ НАЙДЕННЫЕ СЛОВА-ТРИГГЕРЫ
        for match in SPECIAL_PII_RE.findall(text):
            findings['special_pii'].add(match.strip().lower())

        mrz_names = self._extract_names_from_mrz(text)
        anchor_data = self._extract_foreign_pii_by_anchors(text)

        if mrz_names:
            findings['fio'].update(mrz_names)
        elif anchor_data['fio']:
            findings['fio'].update(anchor_data['fio'])
        else:
            findings['fio'].update(self._extract_fios(text))

        if anchor_data['passport']:
            findings['passport'].update(anchor_data['passport'])

        findings['passport'] = self._deduplicate_ids(findings['passport'])
        findings['snils'] = self._deduplicate_ids(findings['snils'])
        findings['inn'] = self._deduplicate_ids(findings['inn'])
        findings['driver_license'] = self._deduplicate_ids(findings['driver_license'])

        counts = {k: len(v) for k, v in findings.items()}
        return {
            'stats': {
                'common_pii': sum(counts[k] for k in ('fio', 'email', 'phone', 'address')),
                'gov_ids': sum(counts[k] for k in ('passport', 'snils', 'inn', 'driver_license')),
                'payment_info': sum(counts[k] for k in ('bank_card', 'bik', 'cvv')),

                # 3. ТЕПЕРЬ ПРОСТО СЧИТАЕМ ДЛИНУ СОБРАННОГО МНОЖЕСТВА
                'special_pii': counts['special_pii'],
            },
            'raw_data': {k: list(v) for k, v in findings.items()}
        }