import sys
import mod_cuda

if __name__ == '__main__':
    device_id = 0

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <grid size> <block size>", file=sys.stderr)
        sys.exit(1)

    try:
        mod_cuda.set_device(device_id)
    except RuntimeError:
        print("Error setting device", file=sys.stderr)
        sys.exit(1)

    try:
        prop = mod_cuda.get_prop(device_id)
    except RuntimeError:
        print("Error getting device properties", file=sys.stderr)
        sys.exit(1)

    try:
        grid_size = mod_cuda.parse_u32(sys.argv[1])
    except ValueError:
        print("Invalid grid size", file=sys.stderr)
        sys.exit(1)
    grid_shape = mod_cuda.Dim3(grid_size, 1, 1)
    try:
        mod_cuda.check_grid_shape(grid_shape, prop)
    except ValueError:
        print("Invalid grid shape", file=sys.stderr)
        sys.exit(1)

    try:
        block_size = mod_cuda.parse_u32(sys.argv[2])
    except ValueError:
        print("Invalid block size", file=sys.stderr)
        sys.exit(1)
    block_shape = mod_cuda.Dim3(block_size, 1, 1)
    try:
        mod_cuda.check_block_shape(block_shape, prop)
    except ValueError:
        print("Invalid block shape", file=sys.stderr)
        sys.exit(1)

    mod_cuda.welcome()
    try:
        mod_cuda.run_kernel(grid_shape, block_shape)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
