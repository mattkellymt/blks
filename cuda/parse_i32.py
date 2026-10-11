import sys
from mod_cuda import parse_i32

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <number>")
        sys.exit(1)
    print(parse_i32(sys.argv[1]))
