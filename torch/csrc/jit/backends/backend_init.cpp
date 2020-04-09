#include <torch/csrc/jit/backends/backend_init.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

namespace torch {
namespace jit {
void initJitBackendBindings(PyObject* module) {
  for (auto& backend : getAllBackends()) {
    // Register custom ops for backend.compile and backend.execute so that the
    // LoweredModule below can call them.
    std::string compile_method_schema =
        backend.name + "::compile(Any a, Dict(str, Any) b) -> Dict(str, Any)";
    std::string execute_method_schema =
        backend.name + "::execute(Any a, Any input) -> Any";

    torch::jit::RegisterOperators reg({
        Operator(
            compile_method_schema,
            [=](Stack& stack) {
              auto method_compile_spec = pop(stack).toGenericDict();
              auto module = pop(stack);
              auto res = backend.instance->compile(module, method_compile_spec);
              push(stack, res);
              return 0;
            },
            c10::AliasAnalysisKind::PURE_FUNCTION),
        Operator(
            execute_method_schema,
            [=](Stack& stack) {
              auto input = pop(stack);
              auto handle = pop(stack);
              auto res = backend.instance->execute(handle, input);
              push(stack, res);
              return 0;
            },
            c10::AliasAnalysisKind::PURE_FUNCTION),
    });

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
          auto preprocessed_module = backend.instance->preprocess(
              cloned_module._ivalue(),
              toIValue(method_compile_spec, any_dict_ty).toGenericDict());

          // Generate LoweredModule.
          Module loweredModule("torch.jit." + backend.name + "LoweredModule");

          // Generate attributes.
          // This is for the method_compile_spec passed in to to_<backend> or
          // loaded from an export model.
          loweredModule.register_attribute(
              "__method_compile_spec",
              any_dict_ty,
              toIValue(method_compile_spec, any_dict_ty).toGenericDict());

          loweredModule.register_attribute(
              "__backend",
              CapsuleType::get(),
              IValue::make_capsule(backend.instance));

          // This is the list of opaque backend handles returned by
          // backend.compile.
          loweredModule.register_attribute(
              "__backend_handles",
              any_dict_ty,
              c10::impl::GenericDict(
                  any_dict_ty->getKeyType(), any_dict_ty->getValueType()));

          // This is the original cloned and preprocessed module.
          loweredModule.register_attribute(
              "__orig_module", AnyType::get(), preprocessed_module);

          // Methods.
          loweredModule.define(R"(
            def __getstate__(self):
                return self.__method_compile_spec, self.__orig_module, self.__backend
            )");

          // This is a convenient wrapper for backend.compile.
          loweredModule.define(
              "\ndef __compile__(self):\n"
              "\tself.__backend_handles = torch.ops." +
              backend.name +
              ".compile(self.__orig_module, self.__method_compile_spec)\n");

          loweredModule.define(R"(
            def __setstate__(self, state):
                self.__method_compile_spec = state[0]
                self.__orig_module = state[1]
                self.__backend = state[2]
                self.__compile__()
            )");

          // This loop generates one method on the LoweredModule for every key
          // in method_compile_spec.
          for (auto& e : method_compile_spec) {
            std::string method_name = py::cast<std::string>(e.first);

            loweredModule.define(
                "\ndef " + method_name + "(self, input):\n\treturn torch.ops." +
                backend.name + ".execute(self.__backend_handles[\"" +
                method_name + "\"], input)\n");
          }

          // Call compile to ensure that the returned Module is ready to run.
          auto state = at::ivalue::Tuple::create(
              toIValue(method_compile_spec, any_dict_ty).toGenericDict(),
              preprocessed_module,
              IValue::make_capsule(backend.instance));
          loweredModule.run_method("__setstate__", state);
          return loweredModule;
        });
  }
}
} // namespace jit
} // namespace torch
