import ctypes
from ctypes import c_char_p, c_float, c_int32, c_uint32, Structure, Union

class Result(Union):
    _fields_ = [
        ("u32", c_uint32),
        ("i32", c_int32),
        ("f32", c_float)
    ]

class Pair(Structure):
    _fields_ = [
        ("u", Result),
        ("ec", c_int32)
    ]

try:
    lib = ctypes.CDLL('./lib_cuda.so')
except OSError as e:
    raise RuntimeError("Failed to load lib_cuda.so") from e

lib.parse_u32.argtypes = [c_char_p]
lib.parse_u32.restype  = Pair

lib.parse_i32.argtypes  = [c_char_p]
lib.parse_i32.restype   = Pair

lib.parse_f32.argtypes  = [c_char_p]
lib.parse_f32.restype   = Pair

def parse_u32(s: str) -> int:
    res = lib.parse_u32(s.encode('utf-8'))
    if res.ec != 0:
        raise ValueError(f"Failed to parse '{s}' as uint32_t (error code: {res.ec})")
    return res.u.u32

def parse_i32(s: str) -> int:
    res = lib.parse_i32(s.encode('utf-8'))
    if res.ec != 0:
        raise ValueError(f"Failed to parse '{s}' as int32_t (error code: {res.ec})")
    return res.u.i32

def parse_f32(s: str) -> float:
    res = lib.parse_f32(s.encode('utf-8'))
    if res.ec != 0:
        raise ValueError(f"Failed to parse '{s}' as float (error code: {res.ec})")
    return res.u.f32
