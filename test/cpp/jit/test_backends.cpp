#include <test/cpp/jit/test_utils.h>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend.h>

namespace torch {
namespace jit {

// This test JIT backend is intended to do the minimal amount of work
// necessary to test that the JIT backend registration endpoints and
// code generation are working correctly. It is not intended to
// produce numerically correct results.
class TestBackend : public PyTorchBackendInterface {
 public:
  explicit TestBackend() {}
  c10::IValue preprocess(
      c10::IValue mod,
      c10::impl::GenericDict method_compile_spec) override {
    return mod;
  }
  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    return method_compile_spec;
  }
  c10::IValue execute(c10::IValue handle, c10::IValue input) override {
    return input;
  }
};

RegisterBackend backend({/*name=*/"test_backend",
                         /*instance=*/c10::make_intrusive<TestBackend>()});

} // namespace jit
} // namespace torch
