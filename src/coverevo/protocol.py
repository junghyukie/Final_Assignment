"""Identity helper used only to resume optional live discovery safely."""
from pathlib import Path
import hashlib


class FrozenProtocolError(RuntimeError):
    pass


def _tree_hash(root: Path, suffixes: tuple[str, ...]) -> str:
    if not root.is_dir():
        raise FrozenProtocolError(f"hash root is not a directory: {root}")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (not suffixes or path.suffix in suffixes)
    )
    if not paths:
        raise FrozenProtocolError(f"hash root has no matching files: {root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()
