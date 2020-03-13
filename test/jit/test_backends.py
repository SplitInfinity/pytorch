import os
import sys
import unittest

import torch
from torch.testing import FileCheck
from torch._six import PY2

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")


class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, x, h):
        new_h = torch.tanh(x + h)
        return new_h


class TestBackends(JitTestCase):
    def test_simple(self):
        # Test compile.
        scripted_module = torch.jit.script(MyModule())
        lowered_module = torch.jit.to_test_backend(scripted_module._c, {"key": "value"})
        
        # Test execute.
        input = torch.randn(5)
        output = lowered_module.forward(input)
        # Test backend always returns input as output.
        self.assertEqual(input, output)

        # Test save and load.
        # lowered_module.save("lowered.pt")
        # loaded_module = torch.jit.load("lowered.pt")
        # loaded_output = loaded_module.forward(input)
        # self.assertEqual(input, loaded_output)
