#include <torch/csrc/jit/backends/backend.h>

namespace torch {
namespace jit {
namespace {
using BackendList = std::vector<Backend>;

// Registry class for backends.
struct BackendRegistry {
 private:
  BackendList backends_;

 public:
  void registerBackend(Backend&& backend) {
    backends_.emplace_back(std::move(backend));
  }

  const BackendList& getAllBackends() const {
    return backends_;
  }
};

BackendRegistry& getRegistry() {
  static BackendRegistry r;
  return r;
}
} // anonymous namespace

RegisterBackend::RegisterBackend(Backend backend) {
  getRegistry().registerBackend(std::move(backend));
}

const std::vector<Backend>& getAllBackends() {
  return getRegistry().getAllBackends();
}
} // namespace jit
} // namespace torch
