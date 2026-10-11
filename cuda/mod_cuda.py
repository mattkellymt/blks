from ctypes import (
    CDLL,
    POINTER,
    Structure,
    byref,
    c_byte,
    c_char_p,
    c_int,
    c_size_t,
    c_uint32,
)
from pathlib import Path

class Dim3(Structure):
    _fields_ = [
        ("x", c_uint32),
        ("y", c_uint32),
        ("z", c_uint32)
    ]

    def __init__(self, x: int = 1, y: int = 1, z: int = 1):
        super().__init__(x, y, z)

    def __repr__(self) -> str:
        return f"Dim3({self.x}, {self.y}, {self.z})"

try:
    lib = CDLL(str(Path(__file__).parent / 'lib_cuda.so'))
except OSError as e:
    raise RuntimeError("Failed to load lib_cuda.so") from e

lib.c_prop_size.argtypes = []
lib.c_prop_size.restype  = c_size_t

class DeviceProp(Structure):
    _fields_ = [
        ("data", c_byte * lib.c_prop_size())
    ]

lib.c_set_device.argtypes = [c_uint32]
lib.c_set_device.restype  = c_int

lib.c_welcome.argtypes = []
lib.c_welcome.restype  = None

lib.c_get_prop.argtypes = [POINTER(DeviceProp), c_uint32]
lib.c_get_prop.restype  = c_int

lib.c_parse_u32.argtypes = [POINTER(c_uint32), c_char_p]
lib.c_parse_u32.restype  = c_int

lib.c_check_grid.argtypes = [Dim3, POINTER(DeviceProp)]
lib.c_check_grid.restype  = c_int

lib.c_check_block.argtypes = [Dim3, POINTER(DeviceProp)]
lib.c_check_block.restype  = c_int

lib.c_run_kernel.argtypes = [Dim3, Dim3]
lib.c_run_kernel.restype  = c_int

lib.c_get_error_string.argtypes = [c_int]
lib.c_get_error_string.restype  = c_char_p

def get_error_string(err: int) -> str:
    return lib.c_get_error_string(err).decode('utf-8')

def set_device(device_id: int) -> None:
    if (err := lib.c_set_device(device_id)) != 0:
        raise RuntimeError(f"Error setting device: {get_error_string(err)}")

def welcome() -> None:
    lib.c_welcome()

def get_prop(device_id: int) -> DeviceProp:
    prop = DeviceProp()
    if (err := lib.c_get_prop(byref(prop), device_id)) != 0:
        raise RuntimeError(f"Error getting device properties: {get_error_string(err)}")
    return prop

def parse_u32(s: str) -> int:
    val = c_uint32()
    if lib.c_parse_u32(byref(val), s.encode('utf-8')) != 0:
        raise ValueError("Failed to parse input as uint32_t")
    return val.value

def check_grid(grid_shape: Dim3, prop: DeviceProp) -> None:
    if lib.c_check_grid(grid_shape, byref(prop)) != 0:
        raise ValueError("Invalid grid shape")

def check_block(block_shape: Dim3, prop: DeviceProp) -> None:
    if lib.c_check_block(block_shape, byref(prop)) != 0:
        raise ValueError("Invalid block shape")

def run_kernel(grid_shape: Dim3, block_shape: Dim3) -> None:
    if (err := lib.c_run_kernel(grid_shape, block_shape)) != 0:
        raise RuntimeError(f"Kernel failed: {get_error_string(err)}")
