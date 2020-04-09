#include <torch/csrc/jit/backends/backend.h>

namespace torch {
namespace jit {
namespace {
using BackendList = std::vector<Backend>;

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

void registerPreprocessor(
    std::string name,
    BackendPreprocessFn fn,
    c10::intrusive_ptr<PyTorchBackendInterface> instance) {
  getRegistry().registerBackend({name, fn, instance});
}

const std::vector<Backend>& getAllBackends() {
  return getRegistry().getAllBackends();
}

} // namespace jit
} // namespace torch
