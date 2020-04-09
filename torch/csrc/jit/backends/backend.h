#pragma once

#include <ATen/core/stack.h>

namespace torch {
namespace jit {

class TORCH_API PyTorchBackendInterface : public torch::CustomClassHolder {
 public:
  PyTorchBackendInterface() {}
  virtual c10::IValue preprocess(
      c10::IValue mod,
      c10::impl::GenericDict method_compile_spec) = 0;
  virtual c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) = 0;
  virtual c10::IValue execute(c10::IValue handle, c10::IValue input) = 0;
};

// This struct represents an external backend that can be used to
// execute JIT subgraphs.
struct TORCH_API Backend {
  Backend() = default;
  /// The name of the backend.
  std::string name;
  /// An instance of the interface object for the backend.
  c10::intrusive_ptr<PyTorchBackendInterface> instance;
};

// Static registration API for backends.
struct TORCH_API RegisterBackend {
  RegisterBackend() = default;
  RegisterBackend(Backend backend);
};

// Get all registered backends.
TORCH_API const std::vector<Backend>& getAllBackends();

} // namespace jit
} // namespace torch
