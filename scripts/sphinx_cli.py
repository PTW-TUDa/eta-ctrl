import subprocess
from pathlib import Path


def build_html() -> None:
    """Run sphinx-build - used as a poetry console script."""
    docs_src = Path("docs")
    docs_out = docs_src / "_build" / "html"
    docs_out.mkdir(parents=True, exist_ok=True)
    cmd = ["sphinx-build", "-b", "html", str(docs_src), str(docs_out)]
    subprocess.run(cmd, check=True)
