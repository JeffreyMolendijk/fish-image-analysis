from setup_dirs import setup_dirs
from rotate_crop import rotate_crop
from measure_magenta import measure_magenta

def main():
    setup_dirs()
    rotate_crop()
    measure_magenta()

if __name__ == '__main__':
    main()
