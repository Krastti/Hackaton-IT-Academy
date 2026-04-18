import re
import difflib
import json
from typing import Dict, List, Any, Optional
import spacy

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

ADDRESS_RE = re.compile(
    r'(?:г\.|город|ул\.|улица|пр-?т\.?|проспект|пер\.|переулок|'
    r'б-?р\.?|бульвар|пл\.|площадь|шоссе|набережная)'
    r'[\s\w\-«»"\'.]{1,100}?(?:д\.|дом)\s*\d{1,5}\w{0,5}',
    re.IGNORECASE
)


def is_luhn_valid(card: str) -> bool:
    digits = [int(d) for d in str(card) if d.isdigit()]
    if len(digits) < 13: return False
    odd_sum = sum(d * 2 - 9 if d * 2 > 9 else d * 2 for d in digits[-2::-2])
    even_sum = sum(digits[-1::-2])
    return (odd_sum + even_sum) % 10 == 0


def is_snils_valid(snils: str) -> bool:
    if len(snils) != 11 or not snils.isdigit(): return False
    num, control = snils[:9], int(snils[9:])
    if int(num) <= 1001998: return True
    checksum = sum(int(d) * (9 - i) for i, d in enumerate(num)) % 101
    return (0 if checksum in (100, 101) else checksum) == control


def normalize_phone(phone: str) -> str:
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 11 and digits.startswith('8'): return '7' + digits[1:]
    return '7' + digits if len(digits) == 10 else digits


class PIIAnalyzer:
    PATTERNS = {
        # ИСПРАВЛЕНИЕ MRZ-КЛОНОВ: Правая граница (?![A-Za-zА-Яа-я0-9]) теперь жестко запрещает
        # откусывать цифры от сплошного текста вроде 9314478268RUS...
        'passport_rf': (
            r'(?i)(?:'
            r'(?:п[аaоo0]сп[оoаa0]рт(?:ные\s+данные)?|серия(?:\s+и\s+номер)?)[:\s#№]{0,10}'
            r'\d{2}[\s\-\.,_]{0,3}\d{2}[\s\-\.,_]{0,3}\d{6}'
            r'|'
            r'(?<![A-Za-zА-Яа-я0-9])(?:\d{4}[\s\-_]*\d{6}|\d{2}[\s\-_]*\d{2}[\s\-_]*\d{6}|\d{10})(?![A-Za-zА-Яа-я0-9])'
            r')'
        ),
        'passport_intl': r'(?:P|I|A|C|V)[A-Z<]{1,2}[A-Z<]{3}[A-Z0-9<]{10,50}',
        'snils': r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}[-\s]\d{2}\b',
        'inn': r'(?i)(?:инн)[:\s#№]{0,10}[\d]{10,12}|\b\d{12}\b',
        'email': r'[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,64}\.[A-Za-z]{2,7}',
        'phone': r'\b(?:\+7|8)[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b',
        'bank_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'driver_license': r'\b\d{2}[\s]?[А-ЯA-Z]{2}[\s]?\d{6}\b',
        'bik': r'(?i)(?:бик|BIK)[:\s]{0,10}\b04\d{7}\b',
        'cvv': r'(?i)(?:cvv2?|cvc2?|csc)[:\s]{0,10}\b\d{3,4}\b',
    }

    def __init__(self):
        try:
            self.nlp = spacy.load('ru_core_news_lg', disable=['parser', 'attribute_ruler'])
        except OSError:
            self.nlp = None

    def _heal_ocr_digits(self, text: str) -> str:
        replacements = {'O': '0', 'о': '0', 'О': '0', 'З': '3', 'з': '3', 'B': '8', 'В': '8', 'I': '1', 'l': '1',
                        'S': '5'}
        for old, new in replacements.items(): text = text.replace(old, new)
        return text

    def _process_match(self, key: str, match: str) -> Optional[str]:
        if key == 'bank_card': return re.sub(r'\D', '', match) if is_luhn_valid(match) else None
        if key == 'snils':
            norm = re.sub(r'\D', '', match)
            return norm if is_snils_valid(norm) else None

        if key == 'passport_rf':
            cleaned = re.sub(r'\D', '', match)
            return cleaned if len(cleaned) == 10 else None
        if key == 'passport_intl':
            cleaned = re.sub(r'[^A-Z0-9<]', '', match.upper())
            return cleaned if len(cleaned) > 15 else None

        if key == 'inn': return re.sub(r'\D', '', match)
        if key == 'phone': return normalize_phone(match)
        if key in ('bik', 'cvv'):
            digits = re.sub(r'\D', '', match)
            return digits if digits else None
        if key in ('email', 'driver_license'): return match.strip()
        return None

    def _extract_fios_spacy(self, text: str) -> set:
        if not self.nlp: return set()
        doc = self.nlp(text[:100000])
        fios = set()
        for ent in doc.ents:
            if ent.label_ == 'PER':
                parts = ent.lemma_.split()
                if len(parts) >= 2 and all(len(p) > 2 for p in parts) and not re.match(r'^[A-Za-z\s]+$', ent.lemma_):
                    fios.add(ent.lemma_.title())
        return fios

    # ИСПРАВЛЕНИЕ ФИО: Бронебойный морфологический поиск (спасает от КАПСА и разрывов строк)
    def _extract_fios_heuristics(self, text: str) -> set:
        fios = set()

        # 1. Жесткая регулярка для русских ФИО (Иванов Иван Иванович)
        # Ищет типичные окончания фамилий (-ов, -ев, -ин) и отчеств (-вич, -вна)
        pattern_fio = r'\b([А-ЯЁа-яё]{2,}(?:ов|ев|ёв|ин|ын|ский|ская|ова|ева|ина|ына|ых|их))\s+([А-ЯЁа-яё]{2,})\s+([А-ЯЁа-яё]{2,}(?:вич|вна|ич|ична|тична))\b'
        for match in re.finditer(pattern_fio, text, re.IGNORECASE):
            fios.add(f"{match.group(1).title()} {match.group(2).title()} {match.group(3).title()}")

        # 2. Поиск по ключевым словам (если ФИО раскидано по строкам)
        f_match = re.search(r'(?i)фамилия[\s\n:\|]+([А-ЯЁа-яёA-Za-z\-]{2,})', text)
        i_match = re.search(r'(?i)имя[\s\n:\|]+([А-ЯЁа-яёA-Za-z\-]{2,})', text)
        o_match = re.search(r'(?i)отчество[\s\n:\|]+([А-ЯЁа-яёA-Za-z\-]{2,})', text)

        if f_match and i_match:
            f = f_match.group(1).title()
            i = i_match.group(1).title()
            o = o_match.group(1).title() if o_match else ""
            fios.add(f"{f} {i} {o}".strip())

        return fios

    def _deduplicate_items(self, items: set, threshold: float = 0.65) -> set:
        sorted_items = sorted(list(items), key=len, reverse=True)
        unique_items = []
        for item in sorted_items:
            is_duplicate = False
            for i, u_item in enumerate(unique_items):
                if difflib.SequenceMatcher(None, item, u_item).ratio() > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_items.append(item)

        mrz_items = [it for it in unique_items if len(it) > 15]
        visual_items = [it for it in unique_items if it not in mrz_items]

        if mrz_items and visual_items: return set(visual_items)
        return set(unique_items)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        actual_keys = set(k if not k.startswith('passport') else 'passport' for k in self.PATTERNS)
        findings = {k: set() for k in actual_keys | {'fio', 'address', 'special_pii'}}

        run_ner = True
        text_for_regex = text

        if text.startswith('{"__is_table_data__":'):
            try:
                table_data = json.loads(text)
                columns = table_data.get("columns", {})
                fio_headers = ['фио', 'fio', 'имя', 'фамилия', 'отчество', 'сотрудник', 'клиент', 'name', 'full name']
                flat_text_parts = []

                for col_name, values in columns.items():
                    if any(h in col_name.lower() for h in fio_headers):
                        for val in values:
                            if len(val.split()) >= 2: findings['fio'].add(val.strip().title())
                    else:
                        flat_text_parts.extend(values)

                text_for_regex = " ".join(flat_text_parts)
                run_ner = False
            except Exception:
                pass

        # === ИЗВЛЕЧЕНИЕ ФИО ===
        if run_ner:
            title_text = re.sub(r'\s+', ' ', text_for_regex).title()
            findings['fio'].update(self._extract_fios_spacy(title_text))

        findings['fio'].update(self._extract_fios_heuristics(text_for_regex))

        # === РЕГУЛЯРКИ И ПАСПОРТА ===
        text_for_digits = self._heal_ocr_digits(text_for_regex)

        for match in re.findall(self.PATTERNS['passport_intl'], text_for_regex):
            processed = self._process_match('passport_intl', match)
            if processed: findings['passport'].add(processed)

        # Вырезаем MRZ зону, чтобы она не фонила
        safe_text_for_regex = re.sub(r'[A-Z0-9<]{25,}', ' ', text_for_regex, flags=re.IGNORECASE)
        safe_text_for_digits = re.sub(r'[A-Z0-9<]{25,}', ' ', text_for_digits, flags=re.IGNORECASE)

        for key, pattern in self.PATTERNS.items():
            if key == 'passport_intl': continue
            target_text = safe_text_for_regex if key in ('email', 'driver_license') else safe_text_for_digits
            for match in re.findall(pattern, target_text):
                save_key = 'passport' if key.startswith('passport') else key
                processed = self._process_match(key, match)
                if processed: findings[save_key].add(processed)

        for match in ADDRESS_RE.findall(safe_text_for_regex):
            findings['address'].add(re.sub(r'\s+', ' ', match).strip())

        for match in SPECIAL_PII_RE.findall(safe_text_for_regex):
            findings['special_pii'].add(match.strip().lower())

        for key in ['passport', 'inn', 'snils', 'phone', 'bank_card']:
            if findings[key]: findings[key] = self._deduplicate_items(findings[key])

        if findings['inn'] and findings['passport']:
            findings['passport'] = findings['passport'] - findings['inn']

        counts = {k: len(v) for k, v in findings.items()}
        return {
            'stats': {
                'common_pii': counts['fio'] + counts['email'] + counts['phone'] + counts['address'],
                'gov_ids': counts['passport'] + counts['snils'] + counts['inn'] + counts['driver_license'],
                'payment_info': counts['bank_card'] + counts['bik'] + counts['cvv'],
                'special_pii': counts['special_pii'],
            },
            'raw_data': {k: list(v) for k, v in findings.items()}
        }