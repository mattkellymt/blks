import sys

from mod_cuda import (
    Dim3,
    check_block,
    check_grid,
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

    set_device(device_id)
    prop = get_prop(device_id)

    grid_size = parse_u32(sys.argv[1])
    grid_shape = Dim3(grid_size, 1, 1)
    check_grid(grid_shape, prop)

    block_size = parse_u32(sys.argv[2])
    block_shape = Dim3(block_size, 1, 1)
    check_block(block_shape, prop)

    welcome()
    run_kernel(grid_shape, block_shape)
