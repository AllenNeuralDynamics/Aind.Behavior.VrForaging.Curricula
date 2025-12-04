import importlib
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

import yaml
from aind_behavior_curriculum.curriculum_utils import export_diagram

if TYPE_CHECKING:
    from aind_behavior_curriculum import Curriculum
else:
    Curriculum = Any


ROOT_DIR = Path(__file__).parent.parent
PACKAGE_NAME = "aind_behavior_vr_foraging_curricula"
SRC_DIR = ROOT_DIR / "src" / PACKAGE_NAME
DOCS_DIR = ROOT_DIR / "docs"
MKDOCS_YML = ROOT_DIR / "mkdocs.yml"
CURRICULA_LABEL = "Curricula"

TO_COPY = ["assets", "examples", "LICENSE"]
log = logging.getLogger("mkdocs")


def on_pre_build(config: Dict[str, Any]) -> None:
    """Mkdocs pre-build hook."""
    for file_or_dir in TO_COPY:
        src: Path = ROOT_DIR / file_or_dir
        dest: Path = DOCS_DIR / file_or_dir

        if src.exists():
            log.info(f"Copying {file_or_dir} to docs...")

            if src.is_file():
                print(f"Copying file {src} to {dest}")
                shutil.copy(src, dest)
            else:
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
            log.info(f"{file_or_dir} copied successfully.")
        else:
            log.warning(f"Source: {file_or_dir} not found, skipping.")

    main()


def render_curricula() -> Dict[str, List[Dict[str, str]]]:
    curricula_structure: Dict[str, List[Dict[str, str]]] = {CURRICULA_LABEL: []}

    for module_dir in [p for p in Path(SRC_DIR).iterdir() if p.is_dir() and not p.name.startswith("_")]:
        module = importlib.import_module(f"{PACKAGE_NAME}.{module_dir.stem}")
        curriculum: Curriculum = getattr(module, "CURRICULUM")

        export_diagram(curriculum, DOCS_DIR / "curricula_diagrams" / f"{module_dir.stem}.svg")

        md_path = DOCS_DIR / f"curricula_diagrams/{module_dir.stem}.md"
        with open(md_path, "w") as f:
            f.write(f"# {module_dir.stem}\n\n")
            f.write(f"**Name**: {curriculum.name}\n\n")
            f.write(f"**Version**: {curriculum.version}\n\n")
            f.write(f"**Pkg-location**: {curriculum.pkg_location}\n\n")
            if class_docs := curriculum.__doc__:
                f.write(f"{class_docs}\n\n")
            f.write("## Diagram\n\n")
            svg_path = f"../curricula_diagrams/{module_dir.stem}.svg"
            f.write(f"![{module_dir.stem} diagram]({svg_path})\n\n")
            json = curriculum.model_dump_json(indent=2)
            f.write("## Specification\n\n")
            f.write(f"```json\n{json}\n```\n")

        curricula_structure[CURRICULA_LABEL].append(
            {module_dir.stem.capitalize(): f"curricula_diagrams/{module_dir.stem}.md"}
        )

    return curricula_structure


def update_mkdocs_yml(curricula_structure: Dict[str, List[Dict[str, str]]]) -> None:
    with open(MKDOCS_YML, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    nav: List[Union[str, Dict[str, Any]]] = config.get("nav", [])
    nav = [item for item in nav if not (isinstance(item, dict) and CURRICULA_LABEL in item)]
    nav.append({CURRICULA_LABEL: curricula_structure[CURRICULA_LABEL]})
    config["nav"] = nav

    with open(MKDOCS_YML, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)


def main() -> None:
    log.info("Regenerating Curricula diagrams...")

    # Generate Curricula structure
    curricula_structure: Dict[str, List[Dict[str, str]]] = render_curricula()

    # Update mkdocs.yml
    update_mkdocs_yml(curricula_structure)

    log.info("Curricula regenerated successfully.")


if __name__ == "__main__":
    main()
