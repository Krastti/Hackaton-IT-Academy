import re
import sys
import difflib
from typing import Dict, List, Any, Optional

import pymorphy3

sys.modules['pymorphy2'] = pymorphy3
sys.modules['pymorphy2.analyzer'] = pymorphy3.analyzer

from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger, Doc


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


_SPECIAL_KW = [
    r'диагноз', r'диабет', r'онкол\w+', r'\bвич\b', r'\bспид\b',
    r'инвалид\w+', r'группа\s+крови', r'анализ\s+крови',
    r'мкб[-\s]?\d+', r'история\s+болезни', r'медицинск\w+\s+карт\w+',
    r'рецепт\b', r'лечени\w+', r'госпитализац\w+', r'хирург\w+',
    r'операци\w+', r'психиатр\w+', r'нарколог\w+', r'иммунодефицит',
    r'отпеч\w+\s+пальц', r'радужн\w+\s+оболочк', r'сетчатк\w+',
    r'голосовой\s+образец', r'биометр\w+', r'дактилоскоп\w+',
    r'распознавание\s+лиц', r'face[\s_-]*id', r'fingerprint',
    r'религи\w+', r'вероисповедан\w+', r'конфесси\w+', r'православ\w+',
    r'мусульман\w+', r'католи\w+', r'атеист\w+', r'буддист\w+',
    r'политическ\w+\s+(?:взгляд|убежден|позиц)\w+',
    r'национальност\w+', r'расов\w+', r'этническ\w+'
]

SPECIAL_PII_RE = re.compile('|'.join(_SPECIAL_KW), re.IGNORECASE)

ADDRESS_RE = re.compile(
    r'(?:г\.|город|ул\.|улица|пр-?т\.?|проспект|пер\.|переулок|'
    r'б-?р\.?|бульвар|пл\.|площадь|шоссе|набережная)'
    r'[\s\w\-«»"\'.]+?(?:д\.|дом)\s*\d+\w*',
    re.IGNORECASE
)


class PIIAnalyzer:
    PATTERNS = {
        # Переименовали в passport, так как теперь ловим любые страны
        'passport_rf': (
            r'(?i)(?:'
            # 1. СНГ/РФ формат с префиксом (ищем строго по вылеченным цифрам)
            r'(?:п[аaоo0]сп[оoаa0]рт(?:ные\s+данные)?|серия(?:\s+и\s+номер)?)[:\s#№]*'
            r'\d{2}[\s\-\.,_]*\d{2}[\s\-\.,_]*\d{6}'
            r'|'
            # 2. РФ формат без префикса (4 цифры + 6 цифр)
            r'(?<!\d)(?:\d{4}[\s\-\.,_]+\d{6}|\d{2}[\s\-\.,_]+\d{2}[\s\-\.,_]+\d{6})(?!\d)'
            r')'
        ),
        'passport_intl': (
            r'(?i)(?:'
            # 3. Универсальная MRZ (Максимально устойчивая к пробелам и мусору от OCR)
            r'(?:P|I|A|C|V)[\s]*[A-Z<KCE«\(]{1,2}[\s]*[A-Z\s]{3,5}[\sA-Z0-9<KCE«\(]{10,}'
            r'|'
            # 4. Иностранные паспорта по английским ключевым словам
            r'(?:passport|document\s*no\.?)[^\w]{0,10}[A-Z0-9]{6,12}\b'
            r')'
        ),
        'snils': r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]\d{2}\b',
        'inn': r'(?i)(?:инн)[:\s#№]*[\d]{10,12}|\b\d{12}\b',
        'email': r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,7}\b',
        'phone': r'\b(?:\+7|8)[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b',
        'bank_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'driver_license': r'\b\d{2}[\s]?[А-ЯA-Z]{2}[\s]?\d{6}\b',
        'bik': r'(?i)(?:бик|BIK)[:\s]*\b04\d{7}\b',
        'cvv': r'(?i)(?:cvv2?|cvc2?|csc)[:\s]*\b\d{3,4}\b',
    }

    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(self.emb)

    def _deduplicate_ids(self, items: set, threshold: float = 0.85) -> set:
        """Слияние похожих идентификаторов и устранение дублей MRZ + Visual."""
        unique_items = []
        for item in items:
            is_duplicate = False
            for u_item in unique_items:
                if difflib.SequenceMatcher(None, item, u_item).ratio() > threshold:
                    # Оставляем более длинный/полный вариант
                    if len(item) > len(u_item):
                        unique_items[unique_items.index(u_item)] = item
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_items.append(item)

        # --- НОВАЯ ЛОГИКА: Устранение двойного подсчета ---
        # Если найден визуальный номер (короткий, из цифр) И строка MRZ (длинная, начинается с P/I/A),
        # они относятся к одному документу в кадре. Оставляем только визуальный номер.
        mrz_items = [it for it in unique_items if re.match(r'(?i)^[PIACV][A-Z]', it) and len(it) > 15]
        visual_items = [it for it in unique_items if it not in mrz_items]

        if mrz_items and visual_items:
            # Нашли и то, и другое -> возвращаем только визуальные номера (схлопываем до 1 паспорта)
            return set(visual_items)

        return set(unique_items)

    def _extract_names_from_mrz(self, text: str) -> set:
        """УРОВЕНЬ 1: Идеальный сценарий. Извлечение ФИО напрямую из машиночитаемой зоны (MRZ)."""
        names = set()
        no_space_text = re.sub(r'[\s\n]+', '', text.upper())

        # Исправленный паттерн: разрешаем буквы сразу после типа документа (для PNRUS)
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
                # Очищаем типичные галлюцинации OCR в MRZ (8->Y, 3->Z, 0->O, 1->I)
                tr_map = str.maketrans('8301', 'YZOI')
                clean_surname = clean_surname.translate(tr_map)
                clean_given = clean_given.translate(tr_map)

                names.add(f"{clean_surname} {clean_given}".title())

        return names

    def _extract_foreign_pii_by_anchors(self, text: str) -> Dict[str, set]:
        """УРОВЕНЬ 2: Якорные слова. Поиск международных паспортов/ID по ключевым полям."""
        anchored_data = {'fio': set(), 'passport': set()}

        # 1. Поиск ФИО
        # Ищем фамилию (после якоря идет минимум 2 буквы)
        surnames = re.findall(r'(?i)\b(?:surname|last name|nom|фамилия)[\s:;\.\|]+([A-ZА-Я]{2,}(?:[- ][A-ZА-Я]{2,})?)',
                              text)
        # Ищем имя
        names = re.findall(
            r'(?i)\b(?:given names?|first name|name|pr[eé]noms?|имя)[\s:;\.\|]+([A-ZА-Я]{2,}(?:[\s-][A-ZА-Я]{2,})?)',
            text)

        # Комбинируем, если нашли и то, и другое
        if surnames and names:
            anchored_data['fio'].add(f"{surnames[0].strip()} {names[0].strip()}".title())
        elif surnames:
            anchored_data['fio'].add(surnames[0].strip().title())
        elif names:
            anchored_data['fio'].add(names[0].strip().title())

        # 2. Поиск номеров документов
        # Ищем 6-14 символов после якорных слов
        doc_nos = re.findall(
            r'(?i)\b(?:document no|passport no|id no|identity no|document number|номер документа)[\s:;\.\|№#]+([A-Z0-9]{6,14})\b',
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
            # Очищаем от слов-префиксов, чтобы вытащить только сам номер/MRZ
            cleaned = re.sub(r'(?i)(паспортные данные|паспорт|серия и номер|серия|passport|document no\.?)', '', match)
            # Оставляем только буквы и цифры (иностранные паспорта содержат буквы!)
            cleaned = re.sub(r'[^A-Za-zА-Яа-я0-9]', '', cleaned)
            # Если после очистки осталось хотя бы 6 символов - это ок.
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
        doc = Doc(clean_ocr_text_for_ner(text))
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
        actual_keys = set(k if not k.startswith('passport') else 'passport' for k in self.PATTERNS)
        findings = {k: set() for k in actual_keys | {'fio', 'address'}}

        text_for_digits = self._heal_ocr_digits(text)

        # 1. Основные регулярные выражения (включая поиск паспортов)
        for key, pattern in self.PATTERNS.items():
            # Иностранные паспорта, email и права ищем по сырому тексту (чтобы не сломать буквы)
            # А passport_rf и остальные госидентификаторы — по вылеченному тексту (text_for_digits)
            target_text = text if key in ('email', 'driver_license', 'passport_intl') else text_for_digits

            for match in re.findall(pattern, target_text):
                # Направляем находки из passport_rf и passport_intl в одну общую корзину 'passport'
                save_key = 'passport' if key.startswith('passport') else key

                processed = self._process_match(save_key, match)
                if processed:
                    findings[save_key].add(processed)

        # 2. Поиск адресов
        for match in ADDRESS_RE.findall(text):
            findings['address'].add(re.sub(r'\s+', ' ', match).strip())

        # =========================================================
        # 3. КАСКАДНЫЙ ПОИСК ФИО И ИНОСТРАННЫХ ДОКУМЕНТОВ
        # =========================================================

        # Уровень 1: MRZ (Машиночитаемая зона - 100% точность)
        mrz_names = self._extract_names_from_mrz(text)

        # Уровень 2: Якорные слова (для иностранных ID без MRZ)
        anchor_data = self._extract_foreign_pii_by_anchors(text)

        # Оркестрация ФИО: Берем самый надежный вариант
        if mrz_names:
            findings['fio'].update(mrz_names)
        elif anchor_data['fio']:
            findings['fio'].update(anchor_data['fio'])
        else:
            # Уровень 3: Нейросеть Natasha (если нет ни MRZ, ни понятных якорей)
            findings['fio'].update(self._extract_fios(text))

        # Дополняем паспорта теми, что нашли по английским якорям
        if anchor_data['passport']:
            findings['passport'].update(anchor_data['passport'])

        # =========================================================
        # 4. Дедупликация OCR-ошибок (чтобы не было 5 паспортов вместо 1)
        # =========================================================
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
                'special_pii': len(SPECIAL_PII_RE.findall(text)),
            },
            'raw_data': {k: list(v) for k, v in findings.items()}
        }