#pragma once

#include <ATen/core/stack.h>

namespace torch {
namespace jit {

// This struct represents an external backend that can be used to
// execute JIT subgraphs.
struct TORCH_API Backend {
  Backend() = default;
  /// The name of the backend.
  std::string name;
  /// A function that should be invoked to convert the JIT graph into
  /// something that the backend can compile.
  std::function<int(Stack&)> preprocess;
  /// A function that should be invoked to compile the JIT graph.
  std::function<int(Stack&)> compile;
  /// A function that should be invoked to execute a previously compiled
  /// JIT subgraph on the backend.
  std::function<int(Stack&)> execute;
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
