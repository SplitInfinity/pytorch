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
        backend.name + "::compile(Any[] a) -> Any[]";
    std::string execute_method_schema =
        backend.name + "::execute(Any[] a) -> Any[]";

    torch::jit::RegisterOperators reg({
        Operator(
            compile_method_schema,
            [=](Stack& stack) {
              auto inputs = pop(stack).toList();
              for (auto i = inputs.begin(), e = inputs.end(); i != e; ++i) {
                push(stack, *i);
              }
              backend.compile(stack);
              return 0;
            },
            c10::AliasAnalysisKind::PURE_FUNCTION),
        Operator(
            execute_method_schema,
            [=](Stack& stack) {
              auto inputs = pop(stack).toList();
              for (auto i = inputs.begin(), e = inputs.end(); i != e; ++i) {
                push(stack, *i);
              }
              backend.execute(stack);
              return 0;
            },
            c10::AliasAnalysisKind::PURE_FUNCTION),
    });

    // Create a method named to_<backend> and pybind it.
    auto m = py::handle(module).cast<py::module>();
    std::string to_backend_method_name = "_jit_to_" + backend.name;
    m.def(
        to_backend_method_name.c_str(),
        [=](Module orig_module, py::dict extra_infos) {
          // TODO: Validate extra_infos.

          // Clone orig_module to make sure backend transformation is
          // functional.
          auto cloned_module = orig_module.clone();

          // Represents of a Type of Dict[str, Any].
          auto any_dict_ty =
              DictType::create(StringType::get(), AnyType::get());

          // Call preprocess.
          Stack stack;
          push(stack, cloned_module._ivalue());
          push(stack, toIValue(extra_infos, any_dict_ty));
          backend.preprocess(stack);
          IValue preprocessed_module = pop(stack);

          // Generate LoweredModule.
          Module loweredModule("torch.jit." + backend.name + "LoweredModule");

          // Generate attributes.
          // This is for the extra_infos passed in to to_<backend> or loaded
          // from an export model.
          loweredModule.register_attribute(
              "__extra_infos", any_dict_ty, toIValue(extra_infos, any_dict_ty));

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
                return self.__extra_infos, self.__orig_module
            )");

          // This is a convenient wrapper for backend.compile.
          loweredModule.define(
              "\ndef __compile__(self):\n"
              //"\tself.__backend_handles.clear()"
              "\thandles: List[Any] = torch.ops." +
              backend.name +
              ".compile(self, self.__orig_module, self.__extra_infos)\n"
              "\tfor h, k in zip(handles, self.__extra_infos.keys()):\n"
              "\t\tself.__backend_handles[k] = h\n");

          loweredModule.define(R"(
            def __setstate__(self, state):
                self.__extra_infos = state[0]
                self.__orig_module = state[1]
                self.__compile__()
            )");

          // This loop generates one method on the LoweredModule for every key
          // in extra_infos.
          for (auto& e : extra_infos) {
            std::string method_name = py::cast<std::string>(e.first);

            loweredModule.define(
                "\ndef " + method_name + "(self, input):\n\treturn torch.ops." +
                backend.name + ".execute(self, self.__backend_handles[\"" +
                method_name + "\"], input)\n");
          }

          // Call compile to ensure that the returned Module is ready to run.
          loweredModule.run_method("__compile__");
          return loweredModule;
        });
  }
}
} // namespace jit
} // namespace torch
