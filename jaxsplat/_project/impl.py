import jax.ffi
from jax.interpreters import mlir, xla
from jax.extend import core

import functools

import _jaxsplat
from jaxsplat._project import lowering, abstract


# register GPU XLA custom calls
# api_version=0 selects XLA's legacy (untyped, opaque-descriptor) custom call
# API, which matches the void(stream, buffers, opaque, opaque_len) kernel
# entry points in lib/ops.cu.
for name, value in _jaxsplat.registrations().items():
    jax.ffi.register_ffi_target(name, value, platform="gpu", api_version=0)


# forward
_project_fwd_p = core.Primitive("project_fwd")
_project_fwd_p.multiple_results = True
_project_fwd_p.def_impl(functools.partial(xla.apply_primitive, _project_fwd_p))
_project_fwd_p.def_abstract_eval(abstract._project_fwd_abs)

mlir.register_lowering(
    prim=_project_fwd_p,
    rule=lowering._project_fwd_rule,
    platform="gpu",
)

# backward
_project_bwd_p = core.Primitive("project_bwd")
_project_bwd_p.multiple_results = True
_project_bwd_p.def_impl(functools.partial(xla.apply_primitive, _project_bwd_p))
_project_bwd_p.def_abstract_eval(abstract._project_bwd_abs)

mlir.register_lowering(
    prim=_project_bwd_p,
    rule=lowering._project_bwd_rule,
    platform="gpu",
)
