#include <torch/csrc/jit/backends/backend_init.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

namespace torch {
namespace jit {
namespace {
// TODO: Delet this, stolen from import_source.cpp.
struct TORCH_API ClassNamespaceValue : public SugaredValue {
  explicit ClassNamespaceValue(c10::QualifiedName name)
      : basename_(std::move(name)) {}

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& name) override {
    auto fullName = c10::QualifiedName(basename_, name);

    if (auto custom_class = getCustomClass(fullName.qualifiedName())) {
      return std::make_shared<ClassValue>(custom_class);
    }

    // If it's none of those things, assume it's another namespace
    return std::make_shared<ClassNamespaceValue>(std::move(fullName));
  }
  std::string kind() const override {
    return "Class Namespace";
  }

 private:
  c10::QualifiedName basename_;
};

// TODO: Delet this.
struct BackendInterfaceResolver : public Resolver {
  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      Function& m,
      const SourceRange& loc) override {
    if (name == "__torch__") {
      return std::make_shared<ClassNamespaceValue>("__torch__");
    }
    return nullptr;
  }

  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    return nullptr;
  }
};

inline std::shared_ptr<BackendInterfaceResolver> backendInterfaceResolver() {
  return std::make_shared<BackendInterfaceResolver>();
}
} // namespace
void initJitBackendBindings(PyObject* module) {
  for (auto& backend : getAllBackends()) {
    std::cout << "Registering backend " << backend.name << "\n";

    // Create a method named to_<backend> and pybind it.
    auto m = py::handle(module).cast<py::module>();
    std::string to_backend_method_name = "_jit_to_" + backend.name;
    m.def(
        to_backend_method_name.c_str(),
        [=](Module orig_module, py::dict method_compile_spec) {
          // TODO: Validate method_compile_spec.

          // Clone orig_module to make sure backend transformation is
          // functional.
          auto cloned_module = orig_module.clone();

          // Represents of a Type of Dict[str, Any].
          auto any_dict_ty =
              DictType::create(StringType::get(), AnyType::get());

          // Call preprocess.
          auto preprocessed_module = backend.preprocess(
              cloned_module._ivalue(),
              c10::impl::toTypedDict<std::string, c10::IValue>(
                  toIValue(method_compile_spec, any_dict_ty).toGenericDict()));

          std::string backend_interface_class =
              "__torch__.torch.classes.backend_interfaces." + backend.name;
          const std::shared_ptr<Resolver> resolver = backendInterfaceResolver();

          // Generate LoweredModule.
          Module loweredModule("torch.jit." + backend.name + "LoweredModule");

          // Generate attributes.
          // This is for the method_compile_spec passed in to to_<backend> or
          // loaded from an export model.
          loweredModule.register_attribute(
              "__method_compile_spec",
              any_dict_ty,
              toIValue(method_compile_spec, any_dict_ty));

          // This is the list of opaque backend handles returned by
          // backend.compile.
          loweredModule.register_attribute(
              "__handles",
              any_dict_ty,
              c10::impl::GenericDict(
                  any_dict_ty->getKeyType(), any_dict_ty->getValueType()));

          // This is the original cloned and preprocessed module.
          loweredModule.register_attribute(
              "__processed_module", AnyType::get(), preprocessed_module);

          loweredModule.register_attribute(
              "__backend",
              OptionalType::create(getCustomClass(backend_interface_class)),
              IValue(c10::nullopt));

          // Methods.
          loweredModule.define(
              R"(
            def __getstate__(self):
                return self.__method_compile_spec, self.__processed_module
            )",
              resolver);

          loweredModule.define(
              "def __setstate__(self, state):\n"
              "\tself.__method_compile_spec = state[0]\n"
              "\tself.__processed_module = state[1]\n"
              "\tself.__backend = __torch__.torch.classes.backend_interfaces." +
                  backend.name +
                  "()\n"
                  "\tbknd = self.__backend\n"
                  "\tassert bknd is not None\n"
                  "\tself.__handles = bknd.compile(self.__processed_module, self.__method_compile_spec)\n",
              resolver);

          // This loop generates one method on the LoweredModule for every key
          // in method_compile_spec.
          for (auto& e : method_compile_spec) {
            std::string method_name = py::cast<std::string>(e.first);

            loweredModule.define(
                "\ndef " + method_name +
                    "(self, input):\n"
                    "\tbackend = __torch__.torch.classes.backend_interfaces." +
                    backend.name +
                    "()\n"
                    "\treturn backend.execute(self.__handles[\"" +
                    method_name + "\"], input)\n",
                resolver);
          }

          // Call compile to ensure that the returned Module is ready to run.
          auto state = at::ivalue::Tuple::create(
              c10::impl::toTypedDict<std::string, IValue>(
                  toIValue(method_compile_spec, any_dict_ty).toGenericDict()),
              preprocessed_module);
          loweredModule.run_method("__setstate__", state);
          return loweredModule;
        });
  }
}
} // namespace jit
} // namespace torch
