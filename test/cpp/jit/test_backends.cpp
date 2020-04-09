#include <test/cpp/jit/test_utils.h>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend.h>

namespace torch {
namespace jit {

class TestBackendInterface : public PyTorchBackendInterface {
 public:
  TestBackendInterface() {}
  c10::Dict<std::string, c10::IValue> compile(
      c10::IValue processed,
      c10::Dict<std::string, c10::IValue> method_compile_spec) override {
    return method_compile_spec;
    // int count = 0;
    // auto typed_spec = c10::impl::toTypedDict<std::string,
    // c10::IValue>(method_compile_spec); c10::impl::GenericDict
    // res(StringType::get(), AnyType::get()); for (auto it =
    // typed_spec.begin(), end = typed_spec.end(); it != end;
    //      ++it) {
    //   res.insert(it->key(), count);
    //   ++count;
    // }

    // return res;
  }

  c10::IValue execute(c10::IValue handle, c10::IValue input) override {
    return input.toTensor();
  }
};

c10::IValue test_backend_preprocess(
    c10::IValue cloned_orig,
    c10::Dict<std::string, c10::IValue> method_compile_spec) {
  return cloned_orig;
}

static auto testBackend = torch::jit::registerBackend<TestBackendInterface>(
    "test_backend",
    test_backend_preprocess);

} // namespace jit
} // namespace torch
