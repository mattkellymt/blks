import sys
import mod_cuda

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <number>")
        sys.exit(1)
    print(mod_cuda.parse_f32(sys.argv[1]))
