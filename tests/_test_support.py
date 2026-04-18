import importlib.util
from pathlib import Path
import shutil
import sys
import types
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_test_env() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    _install_optional_dependency_stubs()


def make_smoke_dir(name: str) -> Path:
    smoke_dir = PROJECT_ROOT / '.codex-smoke' / name
    shutil.rmtree(smoke_dir, ignore_errors=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    return smoke_dir


def print_test_report(title: str, test_cases: Iterable[tuple[str, Callable[[], None]]]) -> None:
    results: list[tuple[str, bool, str]] = []

    for test_name, test_case in test_cases:
        try:
            test_case()
            results.append((test_name, True, 'OK'))
        except Exception as exc:
            results.append((test_name, False, f'Ошибка: {exc}'))

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(title)
    print(f'- Всего тестов: {total}')
    print(f'- Успешно: {passed}')
    print(f'- С ошибками: {total - passed}')
    for test_name, success, message in results:
        status = 'PASSED' if success else 'FAILED'
        print(f'- {test_name}: {status} ({message})')


def _install_optional_dependency_stubs() -> None:
    if importlib.util.find_spec('tqdm') is None:
        tqdm_module = types.ModuleType('tqdm')
        tqdm_module.tqdm = lambda iterable, **kwargs: iterable
        sys.modules['tqdm'] = tqdm_module

    if importlib.util.find_spec('cv2') is None:
        cv2_module = types.ModuleType('cv2')
        cv2_module.COLOR_BGR2GRAY = 0
        cv2_module.COLOR_GRAY2BGR = 0
        cv2_module.COLOR_RGBA2BGR = 0
        cv2_module.COLOR_RGB2BGR = 0
        cv2_module.ADAPTIVE_THRESH_GAUSSIAN_C = 0
        cv2_module.THRESH_BINARY = 0
        cv2_module.ROTATE_90_COUNTERCLOCKWISE = 0
        cv2_module.CAP_PROP_FPS = 0
        cv2_module.CAP_PROP_FRAME_COUNT = 0
        cv2_module.CAP_PROP_POS_FRAMES = 0
        cv2_module.cvtColor = lambda image, code: image
        cv2_module.adaptiveThreshold = lambda *args, **kwargs: args[0]
        cv2_module.equalizeHist = lambda image: image
        cv2_module.rotate = lambda image, code: image
        cv2_module.imread = lambda file_path: None
        cv2_module.VideoCapture = lambda file_path: None
        sys.modules['cv2'] = cv2_module

    if importlib.util.find_spec('docx') is None:
        docx_module = types.ModuleType('docx')
        docx_module.Document = lambda file_path: None
        sys.modules['docx'] = docx_module

    if importlib.util.find_spec('easyocr') is None:
        easyocr_module = types.ModuleType('easyocr')

        class _Reader:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def readtext(self, *args, **kwargs) -> list[str]:
                return []

        easyocr_module.Reader = _Reader
        sys.modules['easyocr'] = easyocr_module

    if importlib.util.find_spec('pandas') is None:
        pandas_module = types.ModuleType('pandas')

        class _ParserError(Exception):
            pass

        class _Errors:
            ParserError = _ParserError

        pandas_module.errors = _Errors()
        pandas_module.read_csv = lambda *args, **kwargs: None
        pandas_module.read_excel = lambda *args, **kwargs: None
        pandas_module.read_parquet = lambda *args, **kwargs: None
        sys.modules['pandas'] = pandas_module

    if importlib.util.find_spec('bs4') is None:
        bs4_module = types.ModuleType('bs4')

        class _BeautifulSoup:
            def __init__(self, raw_text: str, parser: str) -> None:
                self.raw_text = raw_text

            def get_text(self, separator: str = ' ', strip: bool = True) -> str:
                return self.raw_text

        bs4_module.BeautifulSoup = _BeautifulSoup
        sys.modules['bs4'] = bs4_module

    if importlib.util.find_spec('striprtf') is None:
        striprtf_package = types.ModuleType('striprtf')
        striprtf_module = types.ModuleType('striprtf.striprtf')
        striprtf_module.rtf_to_text = lambda text: text
        sys.modules['striprtf'] = striprtf_package
        sys.modules['striprtf.striprtf'] = striprtf_module
