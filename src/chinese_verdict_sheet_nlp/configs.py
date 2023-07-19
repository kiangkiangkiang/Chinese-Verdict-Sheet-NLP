from pathlib import Path

from dotenv import load_dotenv

import chinese_verdict_sheet_nlp

PROJECT_BASE = Path(chinese_verdict_sheet_nlp.__file__).parent
DOTENV_FILE_PATH = PROJECT_BASE / '.env.prod'

load_dotenv(DOTENV_FILE_PATH)
