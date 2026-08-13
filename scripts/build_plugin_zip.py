"""Build an installable QGIS plugin ZIP from the repository checkout."""

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "jap_map"
CORE_PACKAGE = ROOT / "histcontour_core"
DIST = ROOT / "dist"
OUTPUT = DIST / "historical-map-tools-0.2.0.zip"


def main():
    DIST.mkdir(exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for package, destination in ((PACKAGE, "jap_map"), (CORE_PACKAGE, "histcontour_core")):
            for path in sorted(package.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, Path(destination) / path.relative_to(package))
        archive.write(ROOT / "LICENSE", Path("jap_map") / "LICENSE")
    print(OUTPUT)


if __name__ == "__main__":
    main()
