from typing import TYPE_CHECKING, Callable, cast

from synth_xfer._eval_engine import (
    ToEvalAntiRange4_4_4,
    ToEvalAntiRange8_8_8,
    ToEvalAntiRange16_16_16,
    ToEvalAntiRange32_32_32,
    ToEvalAntiRange64_64_64,
    ToEvalKnownBits4_4_4,
    ToEvalKnownBits8_8_8,
    ToEvalKnownBits16_16_16,
    ToEvalKnownBits32_32_32,
    ToEvalKnownBits64_64_64,
    ToEvalSConstRange4_4_4,
    ToEvalSConstRange8_8_8,
    ToEvalSConstRange16_16_16,
    ToEvalSConstRange32_32_32,
    ToEvalSConstRange64_64_64,
    ToEvalUConstRange4_4_4,
    ToEvalUConstRange8_8_8,
    ToEvalUConstRange16_16_16,
    ToEvalUConstRange32_32_32,
    ToEvalUConstRange64_64_64,
    enum_low_antirange_4_4_4,
    enum_low_antirange_8_8_8,
    enum_low_antirange_16_16_16,
    enum_low_antirange_32_32_32,
    enum_low_antirange_64_64_64,
    enum_low_knownbits_4_4_4,
    enum_low_knownbits_8_8_8,
    enum_low_knownbits_16_16_16,
    enum_low_knownbits_32_32_32,
    enum_low_knownbits_64_64_64,
    enum_low_sconstrange_4_4_4,
    enum_low_sconstrange_8_8_8,
    enum_low_sconstrange_16_16_16,
    enum_low_sconstrange_32_32_32,
    enum_low_sconstrange_64_64_64,
    enum_low_uconstrange_4_4_4,
    enum_low_uconstrange_8_8_8,
    enum_low_uconstrange_16_16_16,
    enum_low_uconstrange_32_32_32,
    enum_low_uconstrange_64_64_64,
    enum_mid_antirange_4_4_4,
    enum_mid_antirange_8_8_8,
    enum_mid_antirange_16_16_16,
    enum_mid_antirange_32_32_32,
    enum_mid_antirange_64_64_64,
    enum_mid_knownbits_4_4_4,
    enum_mid_knownbits_8_8_8,
    enum_mid_knownbits_16_16_16,
    enum_mid_knownbits_32_32_32,
    enum_mid_knownbits_64_64_64,
    enum_mid_sconstrange_4_4_4,
    enum_mid_sconstrange_8_8_8,
    enum_mid_sconstrange_16_16_16,
    enum_mid_sconstrange_32_32_32,
    enum_mid_sconstrange_64_64_64,
    enum_mid_uconstrange_4_4_4,
    enum_mid_uconstrange_8_8_8,
    enum_mid_uconstrange_16_16_16,
    enum_mid_uconstrange_32_32_32,
    enum_mid_uconstrange_64_64_64,
    eval_antirange_4_4_4,
    eval_antirange_8_8_8,
    eval_antirange_16_16_16,
    eval_antirange_32_32_32,
    eval_antirange_64_64_64,
    eval_knownbits_4_4_4,
    eval_knownbits_8_8_8,
    eval_knownbits_16_16_16,
    eval_knownbits_32_32_32,
    eval_knownbits_64_64_64,
    eval_sconstrange_4_4_4,
    eval_sconstrange_8_8_8,
    eval_sconstrange_16_16_16,
    eval_sconstrange_32_32_32,
    eval_sconstrange_64_64_64,
    eval_uconstrange_4_4_4,
    eval_uconstrange_8_8_8,
    eval_uconstrange_16_16_16,
    eval_uconstrange_32_32_32,
    eval_uconstrange_64_64_64,
)
from synth_xfer._util.domain import AbstractDomain
from synth_xfer._util.eval_result import EvalResult, PerBitRes, get_per_bit
from synth_xfer._util.jit import Jit
from synth_xfer._util.lower import LowerToLLVM
from synth_xfer._util.parse_mlir import HelperFuncs

if TYPE_CHECKING:
    from synth_xfer._eval_engine import BW, Results, ToEval


def parse_engine_output(output: str) -> list[EvalResult]:
    bw_evals = output.split("---\n")
    bw_evals.reverse()
    per_bits = [get_per_bit(x) for x in bw_evals if x != ""]

    ds: list[list[PerBitRes]] = [[] for _ in range(len(per_bits[0]))]
    for es in per_bits:
        for i, e in enumerate(es):
            ds[i].append(e)

    return [EvalResult(x) for x in ds]


def setup_eval(
    bw: "BW",
    samples: int | None,
    seed: int,
    helper_funcs: HelperFuncs,
    domain: AbstractDomain,
    jit: Jit,
) -> "ToEval":
    lowerer = LowerToLLVM(bw)
    crt = lowerer.add_fn(helper_funcs.crt_func, shim=True)
    op_constraint = (
        lowerer.add_fn(helper_funcs.op_constraint_func, shim=True)
        if helper_funcs.op_constraint_func
        else None
    )

    jit.add_mod(str(lowerer))
    concrete_fn_ptr = jit.get_fn_ptr(crt.name)
    constraint_fn_ptr = jit.get_fn_ptr(op_constraint.name) if op_constraint else None

    # TODO need to fix unary enum and eval
    low_fns: dict[tuple[AbstractDomain, "BW"], Callable[[int, int | None], "ToEval"]] = {
        (AbstractDomain.AntiRange, 4): enum_low_antirange_4_4_4,
        (AbstractDomain.AntiRange, 8): enum_low_antirange_8_8_8,
        (AbstractDomain.AntiRange, 16): enum_low_antirange_16_16_16,
        (AbstractDomain.AntiRange, 32): enum_low_antirange_32_32_32,
        (AbstractDomain.AntiRange, 64): enum_low_antirange_64_64_64,
        (AbstractDomain.UConstRange, 4): enum_low_uconstrange_4_4_4,
        (AbstractDomain.UConstRange, 8): enum_low_uconstrange_8_8_8,
        (AbstractDomain.UConstRange, 16): enum_low_uconstrange_16_16_16,
        (AbstractDomain.UConstRange, 32): enum_low_uconstrange_32_32_32,
        (AbstractDomain.UConstRange, 64): enum_low_uconstrange_64_64_64,
        (AbstractDomain.KnownBits, 4): enum_low_knownbits_4_4_4,
        (AbstractDomain.KnownBits, 8): enum_low_knownbits_8_8_8,
        (AbstractDomain.KnownBits, 16): enum_low_knownbits_16_16_16,
        (AbstractDomain.KnownBits, 32): enum_low_knownbits_32_32_32,
        (AbstractDomain.KnownBits, 64): enum_low_knownbits_64_64_64,
        (AbstractDomain.SConstRange, 4): enum_low_sconstrange_4_4_4,
        (AbstractDomain.SConstRange, 8): enum_low_sconstrange_8_8_8,
        (AbstractDomain.SConstRange, 16): enum_low_sconstrange_16_16_16,
        (AbstractDomain.SConstRange, 32): enum_low_sconstrange_32_32_32,
        (AbstractDomain.SConstRange, 64): enum_low_sconstrange_64_64_64,
    }

    mid_fns: dict[
        tuple[AbstractDomain, "BW"], Callable[[int, int | None, int, int], "ToEval"]
    ] = {
        (AbstractDomain.AntiRange, 4): enum_mid_antirange_4_4_4,
        (AbstractDomain.AntiRange, 8): enum_mid_antirange_8_8_8,
        (AbstractDomain.AntiRange, 16): enum_mid_antirange_16_16_16,
        (AbstractDomain.AntiRange, 32): enum_mid_antirange_32_32_32,
        (AbstractDomain.AntiRange, 64): enum_mid_antirange_64_64_64,
        (AbstractDomain.UConstRange, 4): enum_mid_uconstrange_4_4_4,
        (AbstractDomain.UConstRange, 8): enum_mid_uconstrange_8_8_8,
        (AbstractDomain.UConstRange, 16): enum_mid_uconstrange_16_16_16,
        (AbstractDomain.UConstRange, 32): enum_mid_uconstrange_32_32_32,
        (AbstractDomain.UConstRange, 64): enum_mid_uconstrange_64_64_64,
        (AbstractDomain.KnownBits, 4): enum_mid_knownbits_4_4_4,
        (AbstractDomain.KnownBits, 8): enum_mid_knownbits_8_8_8,
        (AbstractDomain.KnownBits, 16): enum_mid_knownbits_16_16_16,
        (AbstractDomain.KnownBits, 32): enum_mid_knownbits_32_32_32,
        (AbstractDomain.KnownBits, 64): enum_mid_knownbits_64_64_64,
        (AbstractDomain.SConstRange, 4): enum_mid_sconstrange_4_4_4,
        (AbstractDomain.SConstRange, 8): enum_mid_sconstrange_8_8_8,
        (AbstractDomain.SConstRange, 16): enum_mid_sconstrange_16_16_16,
        (AbstractDomain.SConstRange, 32): enum_mid_sconstrange_32_32_32,
        (AbstractDomain.SConstRange, 64): enum_mid_sconstrange_64_64_64,
    }

    if samples:
        return mid_fns[domain, bw](concrete_fn_ptr, constraint_fn_ptr, samples, seed)
    else:
        return low_fns[domain, bw](concrete_fn_ptr, constraint_fn_ptr)


EvalFn = Callable[["ToEval", list[int], list[int]], "Results"]


# TODO may want to just pass whole jit in here
def eval_transfer_func(
    to_eval: "ToEval",
    xfers: list[int],
    bases: list[int],
) -> list[EvalResult]:
    d: dict[type["ToEval"], EvalFn] = {
        ToEvalKnownBits4_4_4: cast(EvalFn, eval_knownbits_4_4_4),
        ToEvalKnownBits8_8_8: cast(EvalFn, eval_knownbits_8_8_8),
        ToEvalKnownBits16_16_16: cast(EvalFn, eval_knownbits_16_16_16),
        ToEvalKnownBits32_32_32: cast(EvalFn, eval_knownbits_32_32_32),
        ToEvalKnownBits64_64_64: cast(EvalFn, eval_knownbits_64_64_64),
        ToEvalAntiRange4_4_4: cast(EvalFn, eval_antirange_4_4_4),
        ToEvalAntiRange8_8_8: cast(EvalFn, eval_antirange_8_8_8),
        ToEvalAntiRange16_16_16: cast(EvalFn, eval_antirange_16_16_16),
        ToEvalAntiRange32_32_32: cast(EvalFn, eval_antirange_32_32_32),
        ToEvalAntiRange64_64_64: cast(EvalFn, eval_antirange_64_64_64),
        ToEvalUConstRange4_4_4: cast(EvalFn, eval_uconstrange_4_4_4),
        ToEvalUConstRange8_8_8: cast(EvalFn, eval_uconstrange_8_8_8),
        ToEvalUConstRange16_16_16: cast(EvalFn, eval_uconstrange_16_16_16),
        ToEvalUConstRange32_32_32: cast(EvalFn, eval_uconstrange_32_32_32),
        ToEvalUConstRange64_64_64: cast(EvalFn, eval_uconstrange_64_64_64),
        ToEvalSConstRange4_4_4: cast(EvalFn, eval_sconstrange_4_4_4),
        ToEvalSConstRange8_8_8: cast(EvalFn, eval_sconstrange_8_8_8),
        ToEvalSConstRange16_16_16: cast(EvalFn, eval_sconstrange_16_16_16),
        ToEvalSConstRange32_32_32: cast(EvalFn, eval_sconstrange_32_32_32),
        ToEvalSConstRange64_64_64: cast(EvalFn, eval_sconstrange_64_64_64),
    }
    res = d[type(to_eval)](to_eval, xfers, bases)

    return parse_engine_output(str(res))
