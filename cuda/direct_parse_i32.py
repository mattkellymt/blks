import sys
import direct_cuda

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <number>")
        sys.exit(1)
    print(direct_cuda.parse_i32(sys.argv[1]))
