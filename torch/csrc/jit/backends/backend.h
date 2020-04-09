#pragma once

#include <ATen/core/stack.h>
#include <torch/custom_class.h>

namespace torch {
namespace jit {

class TORCH_API PyTorchBackendInterface : public torch::CustomClassHolder {
 public:
  PyTorchBackendInterface() {}
  virtual c10::Dict<std::string, c10::IValue> compile(
      c10::IValue processed,
      c10::Dict<std::string, c10::IValue> method_compile_spec) = 0;

  virtual c10::IValue execute(c10::IValue handle, c10::IValue input) = 0;
};

using BackendPreprocessFn = std::function<
    c10::IValue(c10::IValue, c10::Dict<std::string, c10::IValue>)>;

struct TORCH_API Backend {
  Backend() = default;
  /// The name of the backend.
  std::string name;
  /// The function to use for preprocessing.
  BackendPreprocessFn preprocess;
  c10::intrusive_ptr<PyTorchBackendInterface> instance;
};

TORCH_API void registerPreprocessor(
    std::string name,
    BackendPreprocessFn fn,
    c10::intrusive_ptr<PyTorchBackendInterface> instance);

TORCH_API const std::vector<Backend>& getAllBackends();

template <class BackendInterfaceClass>
torch::class_<BackendInterfaceClass> registerBackend(
    std::string name,
    BackendPreprocessFn fn) {
  auto cls = torch::class_<BackendInterfaceClass>("backend_interfaces", name)
                 .def(torch::init<>())
                 .def("compile", &BackendInterfaceClass::compile)
                 .def("execute", &BackendInterfaceClass::execute);
  registerPreprocessor(name, fn, c10::make_intrusive<BackendInterfaceClass>());
  return cls;
}

} // namespace jit
} // namespace torch
