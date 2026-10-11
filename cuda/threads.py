import sys
import mod_cuda

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <grid size> <block size>", file=sys.stderr)
        sys.exit(1)

    try:
        grid_size = mod_cuda.parse_u32(sys.argv[1])
    except ValueError:
        print("Invalid grid size", file=sys.stderr)
        sys.exit(1)

    try:
        block_size = mod_cuda.parse_u32(sys.argv[2])
    except ValueError:
        print("Invalid block size", file=sys.stderr)
        sys.exit(1)

    sys.exit(mod_cuda.launch_threads(grid_size, block_size))
