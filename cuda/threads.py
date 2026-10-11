import sys
from mod_cuda import (
    check_block_shape,
    check_grid_shape,
    Dim3,
    get_prop,
    parse_u32,
    run_kernel,
    set_device,
    welcome,
)

if __name__ == '__main__':
    device_id = 0

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <grid size> <block size>", file=sys.stderr)
        sys.exit(1)

    try:
        set_device(device_id)
    except RuntimeError:
        print("Error setting device", file=sys.stderr)
        sys.exit(1)

    try:
        prop = get_prop(device_id)
    except RuntimeError:
        print("Error getting device properties", file=sys.stderr)
        sys.exit(1)

    try:
        grid_size = parse_u32(sys.argv[1])
    except ValueError:
        print("Invalid grid size", file=sys.stderr)
        sys.exit(1)
    grid_shape = Dim3(grid_size, 1, 1)
    try:
        check_grid_shape(grid_shape, prop)
    except ValueError:
        print("Invalid grid shape", file=sys.stderr)
        sys.exit(1)

    try:
        block_size = parse_u32(sys.argv[2])
    except ValueError:
        print("Invalid block size", file=sys.stderr)
        sys.exit(1)
    block_shape = Dim3(block_size, 1, 1)
    try:
        check_block_shape(block_shape, prop)
    except ValueError:
        print("Invalid block shape", file=sys.stderr)
        sys.exit(1)

    welcome()
    try:
        run_kernel(grid_shape, block_shape)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
