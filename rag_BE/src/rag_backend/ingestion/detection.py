from pathlib import Path

from rag_backend.ingestion.contracts import PreprocessFailure


def detect(path: Path, filename: str) -> str:
    extension = Path(filename).suffix.lower()
    with path.open('rb') as source:
        header = source.read(1024)
    if extension not in {'.pdf', '.txt', '.md'}:
        raise PreprocessFailure('unsupported_file_type')
    is_pdf = header.lstrip().startswith(b'%PDF-')
    if (extension == '.pdf') != is_pdf:
        raise PreprocessFailure('file_type_mismatch')
    return extension[1:]
