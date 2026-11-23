"""Bundle the single-source Markdown guides in installed wheels."""

from pathlib import Path
from shutil import copyfile

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithGuides(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        target = Path(self.build_lib) / "agentinputlab" / "_bundled_docs"
        for source in sorted((root / "docs").rglob("*.md")):
            destination = target / source.relative_to(root / "docs")
            destination.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source, destination)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        target = Path(self.build_lib) / "agentinputlab" / "_bundled_docs"
        return outputs + [str(path) for path in sorted(target.rglob("*.md"))]


setup(cmdclass={"build_py": BuildWithGuides})
