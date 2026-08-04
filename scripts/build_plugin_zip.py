"""Build an installable QGIS plugin ZIP from the repository checkout."""

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "jap_map"
DIST = ROOT / "dist"
OUTPUT = DIST / "historical-map-tools-0.1.0.zip"


def main():
    DIST.mkdir(exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, Path("jap_map") / path.relative_to(PACKAGE))
        archive.write(ROOT / "LICENSE", Path("jap_map") / "LICENSE")
    print(OUTPUT)


if __name__ == "__main__":
    main()
