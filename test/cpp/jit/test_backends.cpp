#include <test/cpp/jit/test_utils.h>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend.h>

namespace torch {
namespace jit {

// This test JIT backend is intended to do the minimal amount of work
// necessary to test that the JIT backend registration endpoints and
// code generation are working correctly. It is not intended to
// produce numerically correct results.
RegisterBackend backend({
    "test_backend",
    /*preprocess=*/
    [](Stack& stack) -> int {
      auto extra_infos = pop(stack).toGenericDict();
      auto mod = pop(stack).toModule();
      // Return the given module.
      push(stack, mod._ivalue());
      return 0;
    },
    [/*compile=*/](Stack& stack) -> int {
      auto extra_infos = pop(stack).toGenericDict();
      auto orig_mod = pop(stack).toModule();
      auto mod = pop(stack).toModule();
      auto handles = c10::impl::GenericList(IntType::get());
      // Use the integer 1 as a handle.
      handles.emplace_back(1);
      push(stack, handles);
      return 0;
    },
    /*execute=*/
    [](Stack& stack) -> int {
      auto input = pop(stack).toTensor();
      auto handle = pop(stack).toInt();
      (void)handle;
      auto mod = pop(stack).toModule();
      // Return the input Tensor as the output.
      push(stack, input);
      return 0;
    },
});

} // namespace jit
} // namespace torch
