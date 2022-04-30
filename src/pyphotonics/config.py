"""store configuration
"""

__all__ = ["PATH"]

import pathlib

home = pathlib.Path.home()
cwd = pathlib.Path.cwd()
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent


class Path:
    module = module_path
    repo = repo_path
    examples = module / "examples"
    example_autoroute = examples / "autoroute.gds"


PATH = Path()

if __name__ == "__main__":
    print(PATH)
