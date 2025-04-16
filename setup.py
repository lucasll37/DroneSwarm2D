from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Diretório único onde tudo será gerado
BUILD_LIB = os.path.join(os.path.dirname(__file__), "build", "lib")
BUILD_TEMP = os.path.join(os.path.dirname(__file__), "build", "temp")

extensions = [
    Extension("sandbox", ["src/sandbox.pyx"]),
    # adicione outros módulos aqui...
]

setup(
    name="drone_swarm2d",
    ext_modules=cythonize(
        extensions,
        build_dir=BUILD_TEMP,      # onde os .c intermediários vão ficar
        compiler_directives={"language_level": "3"},
    ),
    # para garantir que o install_subcommand respeite o build_lib:
    options={
        "build_ext": {
            "build_lib": BUILD_LIB,
            "build_temp": BUILD_TEMP,
            "inplace": False,
        }
    }
)
