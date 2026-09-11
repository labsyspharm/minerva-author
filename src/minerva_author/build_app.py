import os
import pathlib
import PyInstaller.__main__


def main():
    base_path = pathlib.Path(__file__).parent.parent.parent
    os.chdir(base_path)
    PyInstaller.__main__.run(['minerva-author.spec', '--noconfirm'])


if __name__ == '__main__':
    main()
