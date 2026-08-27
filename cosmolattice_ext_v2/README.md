# Thermal-inflation extension for CosmoLattice v2.0

Port of `cosmolattice_ext/` (CosmoLattice v1.1.2) to
[CosmoLattice v2.0.0](https://arxiv.org/abs/2607.24978), living in
`external/cosmolattice_v2`.

The v1 tree is **untouched**: `external/cosmolattice` stays pinned at
v1.1.2-30-ga5654d6, `cosmolattice_ext/` and `simulation/run_cosmolattice.py`
still build and run exactly as before, and a snapshot of both sits in
`backups/`. Everything v2 is suffixed `_v2`, including the run directories
(`..._CL_v2`), so v1 and v2 results never collide.

| | v1 | v2 |
|---|---|---|
| code | `external/cosmolattice` | `external/cosmolattice_v2` |
| headers | `cosmolattice_ext/` | `cosmolattice_ext_v2/` |
| model / binary | `thermal_inflation` | `thermal_inflation_v2` |
| evolver keyword | `stochasticrk` | `stochastic` |
| driver | `simulation/run_cosmolattice.py` | `simulation/run_cosmolattice_v2.py` |

## Quick start

```bash
python tools/export_thermal_splines.py          # once: writes the J-integral tables
git clone --branch v2.0.0 https://github.com/cosmolattice/cosmolattice.git \
    external/cosmolattice_v2
python simulation/run_cosmolattice_v2.py --install --build --jobs 8 \
    --Nx 64 --T0 1230 --tMax 400 --param_set set8 \
    --n_scalars 2 --with_gws --measure_defects --expansion_mode staged
```

`--install` symlinks the headers and applies the upstream registrations
(idempotent, all inside `// >>> thermal-inflation` markers). `--build` runs
CMake, which fetches TempLat and Kokkos itself.

**Compiler.** TempLat v1.0.0 uses CTAD on an alias template (C++20, P1814R0).
Apple clang 15 rejects it with `error: alias template 'complex' requires
template arguments`; the build defaults to Homebrew `g++-16`. Override with
`--cxx` or `CL2_CXX`. `--hdf5` is needed for field snapshots.

## Files

| File | Role |
|------|------|
| `models/thermal_tables.hpp` | Unchanged from v1: loader/evaluator of `V`, `V'`, `V''` from the binary J-tables. No CosmoLattice dependencies. |
| `models/thermal_force.h` | TempLat operators wrapping the per-site table lookup. Ported to the v2 `eval()` / `DoEval::eval` operator API. |
| `models/thermal_inflation.h` | The model. Same physics as v1; every raw per-site loop is now a lattice expression. |
| `evolvers/stochasticrk.h` | `StochasticLangevin`: friction + FDT noise; `ou`, `verlet`, `numba`, `fused_rk2`. |
| `parameter-files/*.in` | Set B / Set C / smoke-test inputs. |
| `tests/fdt_check.py` | Reproduces the fluctuation-dissipation tables in §1, for v2 and (with `--with-v1`) v1. |
| `patches/cosmolattice_v2.0.0_registration.patch` | The upstream edits, as a reviewable diff. |

Dropped from v1: `measurements/field_snapshot.hpp`. v2 writes field snapshots
itself in HDF5 (`snapshots = S`), with subvolume and stride support, so the
custom `field_states/*.raw` writer is gone along with the v1 `--steps`
snapshot plumbing.

## 1. The stochastic noise: what changed and why

### The v2 backend forced a rewrite

TempLat v1.0.0 keeps fields in Kokkos views and removed the per-site access API
(`itX()`, `Field::get(ptrdiff_t)`, `getSet(ptrdiff_t)`). The v1 default scheme,
`stochastic_scheme = numba`, was a hand-written 4-pass RK2 over raw site indices
that reproduced `latticeSimeRescale_numba.rk2_fused` draw-by-draw, including its
Box–Muller hash RNG. That cannot be transcribed literally, so v2 uses
`RandomGaussianFieldConfig`: a counter-based per-site generator keyed on the
**global** coordinate, which gives decomposition-independent white noise and
re-draws on every assignment (the expression machinery bumps its generation
counter through `preGet`/`postGet`). Numba parity, if wanted, has to be
re-established statistically against v1 rather than draw-by-draw.

### Two coefficient bugs, found while validating

Writing the port from scratch made it worth checking the Langevin coefficients
against the CosmoLattice program-variable convention. CL evolves
`pi = a^(3-alpha) dphiTilde/dEta` with `dEta = a^(-alpha) omegaStar dt`, so
`pi = a^3 phidot / (fStar omegaStar)` whatever `alpha` is. Pushing

```
phiddot + (3H + eta) phidot - Lap phi / a^2 + V' = xi,
Var[d phidot] = 2 eta T dt / (a^3 dx_com^3)
```

through that change of variables gives

```
friction  :  d pi / dEta  ⊃  - a^alpha (eta/omegaStar) pi
noise     :  Var[d pi]     =  2 a^(3+alpha) (eta/omegaStar) T dEta / (dx_com^3 fStar^2 omegaStar^2)
equilibrium: <pi^2>_eq     =  a^3 T / (dx_com^3 fStar^2 omegaStar^2)
```

Two things follow that the first draft of the port got wrong:

- **The `1/omegaStar^2` is not optional.** Without it the injected noise is too
  large by `omegaStar^2 = mphi^2 = 1e6` in variance. v1 has this factor; the
  port had dropped it.
- **The `3H` drag must not be added by hand.** It is already inside `pi`'s
  `a^3`, exactly as for every stock CL evolver. Adding it again double counts.
  Numerically irrelevant here (`3H/eta ~ 5e-5`), but it was wrong.

Both are fixed. The `a`-exponents were also corrected from `1/a^3` to `a^3`,
which matters only over many e-folds (`a` moves ~1% in a typical run).

### `ou` is now the default, and `verlet` is a diagnostic

| `stochastic_scheme` | friction | sampled temperature |
|---|---|---|
| `ou` (default) | exact Ornstein–Uhlenbeck half-steps, Strang-split around a plain Verlet step: `pi <- c pi + sqrt((1-c^2) <pi^2>_eq) z`, `c = exp(-eta dt/2)` | exactly `T`, at any `eta*dt` |
| `verlet` | explicit, folded into each half kick (the v1 "fused" scheme) | `T / (1 - eta*dt/4)`, unstable above `eta*dt = 2` |
| `numba` | same Verlet integrator, noise × `1/√2` (legacy amplitude parity) | `~0.61 T` at production `dt` |
| `fused_rk2` | 4-pass predictor–corrector RK2 + two half-kicks of `0.5 σ_full` (Numba staging) | `~0.61 T` at production `dt` |

Verified on a 32³ non-expanding box at fixed `T = eta_phys = 7350 GeV`,
`mphi = 1000`, `dx_phys = 1e-3`, `fStar = 1e15`, where
`<pi^2>_eq = 7.35e-24`. Measured `<pi^2>/<pi^2>_eq`:

| `dt` | `eta*dt` | `ou` | `verlet` | `1/(1 - eta*dt/4)` |
|---|---|---|---|---|
| 0.0125 | 0.09 | 1.009 | 1.033 | 1.024 |
| 0.025 | 0.18 | 1.001 | 1.049 | 1.048 |
| 0.05 | 0.37 | 1.000 | 1.101 | 1.101 |
| 0.10 | 0.74 | 1.008 | 1.238 | 1.225 |
| 0.20 | 1.47 | 0.990 | 1.666 | 1.581 |
| 0.30 | 2.21 | 1.068 | 5.84 | — |
| 0.50 | 3.68 | 9.88 | 3480 | — |

Reproduce with `python cosmolattice_ext_v2/tests/fdt_check.py`. `ou` tracks the
exact answer to sub-percent up to `dt = 0.2`; `verlet` follows the analytic
explicit-friction bias to three digits. Both fail at `dt >= 0.3`,
but for a shared reason that has nothing to do with the noise: the conservative
Verlet core cannot resolve `m_eff/mphi ~ 8` at that step size. **Keep
`dt <= 0.2`** for these parameters.

### Consequences for the existing v1 runs

The production v1 inputs use `dt = 0.1`, `eta_phys = T0 = 7350 GeV`,
`mphi = 1000`, i.e. `eta*dt = 0.735`. Running the same box through the v1 binary
(`tests/fdt_check.py --with-v1`) gives:

| scheme | measured `<pi^2>/<pi^2>_eq` | predicted |
|---|---|---|
| v1 `numba` (v1 default) | **0.616** | 0.613 |
| v1 `fdt` | 1.233 | 1.225 |
| v2 `verlet`, same `dt` | 1.238 | 1.225 |
| v2 `ou` | 1.008 | 1 |

Two independent effects, both now understood:

1. `numba` injects **half** the FDT variance by construction — that is what the
   `sqrt(2)` in v1's `fdt` option restores, and it is inherited from the numba
   reference, not a coding slip. v1 `fdt` and v2 `verlet` agree to 1.3%, which
   cross-validates the two independent implementations.
2. Explicit friction at `eta*dt = 0.735` adds a further +22%.

Net: **the v1 default runs sample `T_eff ≈ 0.61 T`.** Since the nucleation rate
depends exponentially on `S_3/T`, this biases `T_c1` and every downstream GW
amplitude. It is a systematic offset, not a broken simulation — but v1 and v2
numbers are not directly comparable, and a v1↔v2 `T_c1` comparison needs
`--stochastic_scheme fdt` on the v1 side.

### `fused_rk2`: Numba staging on the v2 backend

`stochastic_scheme = fused_rk2` (aliases `rk2`, `numba_rk2`) is an expression-based
4-pass predictor–corrector RK2 with the same half-noise construction as Numba /
v1 `numbaRK2` (`0.5 * σ_full` on each of two independent corrector kicks). Scale
factor is held fixed across the two half-steps and advanced once at the end.
This is the structural match to the old fused RK2; it is still not bit-identical
(different RNG). Prefer `fused_rk2` when comparing integrator *shape* to Numba;
keep `numba` only for continuity with earlier v2 runs that used Verlet+½ FDT.

## 2. v2 cosmic defects and GWs in the TIPT scenario

### Defects: usable, and it replaces a post-processing step

Setting `DefectsModel = true` in `ModelPars` with `Ns == 2, NCs == 0` makes
`DefectsMeasurer` emit, into `average_defects.txt`:

- `winfindLengthStrings` — total comoving string length from the plaquette
  winding numbers of `Complexify(fldS0, fldS1)`, the observable described in
  §3.3 of the paper;
- weighted `Ekin/Egrad/Epot/Etot/Elag` restricted to the defect cores;

plus `average_norm.txt` with `<|phi|>`, `<|phi|^2>` and its variance, and a `norm`
power spectrum. Enable with `--measure_defects` (requires `--n_scalars 2`; the
winding is meaningless for a single real component, and the driver refuses it).

This is the in-simulation replacement for `tools/compute_strings_cl.py`, which
had to re-derive windings from exported snapshots. Verified working end to end
on a 32³ staged run: string length falls from the white-noise value 21625 at
`t = 0` to numerical zero once the field settles in the true vacuum.

Two upstream guards block `DefectsModel` for us and are patched off **for this
model only** (both keyed on `requires { m.maybeDisableLangevin(); }`):

- `scalefactorinitializer.h` rejects defects with self-consistent expansion,
  because (extra)fattening is untested there. We never fatten, and our scale
  factor follows the model's prescribed `H(T)` rather than the lattice energy.
  Upstream explicitly sanctions commenting the exception out.
- `scalarsingletinitializer.h` demands `ICtype_S` be a scaling network or
  defect white noise. Our strings condense out of the thermal bath and the
  model overwrites the ICs immediately after `initialize()` anyway.

**What we deliberately do not use** from the v2 defect module: the scaling
network ICs of Eq. (35), the diffusion pre-phase, and fattening / extra-fattening.
All three exist to reach the scaling regime quickly for a network that formed in
the far past. Our strings are a *product* of the transition we are simulating,
so their formation is the physics — pre-seeding a scaling network would destroy
it. If the scaling-regime tail ever becomes the object of interest, that is when
those tools become relevant.

### GWs: same physics, cheaper, same normalization caveat

§4.2 replaces the 6 unphysical `u_ij` of v1 with 5 traceless `v_ij`
(`v_33 = -v_11 - v_22`), reconstructing `h_ij` by TT projection at measurement
time. That is a ~17% memory saving in the GW sector and no change in physics.
`doLFforGWs` (default true) also lets the GW sector run leapfrog independently
of the matter evolver; our `StochasticLangevin` bypasses that and drives its own
kick–drift–kick GW sub-step with the same `dt`, since the thermal noise must
never reach the GW sector. Confirmed producing `average_energies_gws.txt` and
`spectra_energy_gws.txt` under the Langevin evolver.

The normalization caveat from v1 is unchanged and is now explicit in the paper:
column 2 of `average_energies_gws.txt` is
`Omega_GW = (1/rho_tot) drho_GW/dlogk` with `rho_tot` the **lattice** energy
density (`Energies::rho(model)`), which equals `rho_c` only under
self-consistent expansion. We prescribe `H` from the background (vacuum +
radiation) while the lattice holds only the scalar, so `rho_tot != rho_c` and the
`rho_tot/rho_c` correction in `postprocess/plot_cl_gw_spectrum.py` is still
required. Column 3 (`rhoGW`) is absolute and needs no correction.

Also newly available and relevant: `PS_version`/`deltaKBin` arbitrary binning
and unbinned spectra (§4.3), useful for resolving the GW peak on small lattices,
and external initial spectra (§4.4).

## 3. Model port notes

Everything below is mechanical; the physics is identical to v1.

- The constructor takes `auto toolBox` instead of
  `std::shared_ptr<MemoryToolBox>` (the one-line change of §5).
- `ParameterParser::get` returns a lazy `ParameterGetter<T>`. Bind it to a
  concrete local before comparing or doing arithmetic:
  `const std::string potType = parser.get<std::string>("potential_type", ...)`,
  otherwise `operator==` and `std::min`/`std::max` fail to resolve.
- Raw loops became lattice expressions, which is what makes them MPI- and
  GPU-correct by construction:
  - numba-style white-noise IC → `RandomGaussianFieldConfig`
  - cubic bubble seed → `SpatialCoordinate` + a product of `heaviside` masks
  - false-vacuum fraction → `average(heaviside(...))`, which already performs
    the global MPI reduction (v1 needed a hand-written `MPI_Allreduce`)
- Model-specific ICs are applied from `source/cosmolattice.cpp` after
  `initializer.initialize(...)`, guarded by
  `if constexpr (requires { model.applyNumbaInitialConditions(); })` so stock
  models are unaffected.

## Upstream edits

Five files, all reversible, all recorded in
`patches/cosmolattice_v2.0.0_registration.patch` and re-applied idempotently by
`--install`:

| File | Marker tag | Why |
|---|---|---|
| `include/CosmoInterface/evolvers/evolvertype.h` | `evolver-enum`, `evolver-parse`, `evolver-string` | register the `STOCHASTIC` evolver keyword |
| `include/CosmoInterface/evolvers/evolver.h` | `include`, `concept`, `construct`, `evolve-dispatch`, `sync-dispatch`, `member` | dispatch to `StochasticLangevin`, gated on the `IsLangevinModel` concept |
| `source/cosmolattice.cpp` | `initial-conditions` | model-specific IC hooks |
| `include/CosmoInterface/initializers/scalefactorinitializer.h` | `defects-expansion` | allow defects with prescribed `H` |
| `include/CosmoInterface/initializers/scalarsingletinitializer.h` | `defects-ic` | allow defects with thermal ICs |

## Open items

- No MPI-decomposition test of `RandomGaussianFieldConfig` yet. It is documented
  as keyed on the global coordinate and the single-rank statistics are exact,
  but a 2-rank run should be checked before any production MPI job.
- No v1↔v2 `T_c1` comparison yet. Given §1 it must be run against v1
  `--stochastic_scheme fdt`, not the v1 default, or the 0.61 `T_eff` offset will
  swamp the comparison.
- v1's `numba` scheme has no v2 equivalent and is not planned.
