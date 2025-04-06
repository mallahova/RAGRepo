from pathlib import Path

EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".cs": Language.CSHARP,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".php": Language.PHP,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".md": Language.MARKDOWN,
    ".html": Language.HTML,
    ".tex": Language.LATEX,
    ".proto": Language.PROTO,
    ".sol": Language.SOL,
    ".cbl": Language.COBOL,
    ".lua": Language.LUA,
    ".pl": Language.PERL,
    ".hs": Language.HASKELL,


def detect_language_from_path(path: str) -> Language:
    ext = Path(path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, Language.MARKDOWN)
