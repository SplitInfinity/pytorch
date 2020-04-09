import torch._C


def to_test_backend(module, extra_info):
    return torch._C._jit_to_test_backend(module, {"forward": extra_info})
