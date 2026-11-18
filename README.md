# ccgen — generating the CCD equations from second quantization

Coupled-cluster equations get long quickly. The CCD amplitude equation, ten terms in the textbook form, becomes 125 separate `einsum` terms once written out, and CCSD and beyond grow much faster. ccgen derives them the way you would on paper, contraction by contraction with Wick's theorem, and writes them out as NumPy code that you can run and check.

**Coming next:**
- the CCSD equations, then CCSDT and EOM-CC (the energy expression and the factors for several T₁ and T₂ are already in place);
- plugging the generated equations into the solvers of [hf_to_cc](https://github.com/Heisenbergnox/hf_to_cc);
- faster generated code: better contraction orders and shared intermediates instead of one `einsum` per term.

## What's inside

- `ccgen.py`: the generator. Each part of the residual ⟨Φ_ij^ab| H̄ |Φ₀⟩ is a list of normal-ordered operators (`ccd_parts`). For each list the generator
  1. finds the allowed, connected full contractions;
  2. takes each sign from the number of crossing lines;
  3. renames the summed indices and attaches integrals, amplitudes and coefficients;
  4. brings each term to a standard form and merges equal ones;
  5. writes one `einsum` line per term.
- `ccd_residual.py`: the generated residual `residual(f, g, t)`, 125 terms. `python provided.py` regenerates it.
- `ccd.py`: `ccd(R, f, g, N)`, which iterates any residual function from MP2 amplitudes.

Water in STO-3G converges in 29 iterations to E_CCD = −0.0506409699 Eh. On 18 molecule/basis runs, the result agrees with the hand-written CCD of hf_to_cc to 1e-9 Eh.

## Usage

Python 3 and NumPy. The integrals here come from [hf_to_cc](https://github.com/Heisenbergnox/hf_to_cc). The arrays use spin orbitals with the `N` occupied ones first, `g[p, q, r, s] = ⟨pq‖rs⟩` and `t[i, j, a, b] = t_ij^ab`.

```python
import numpy as np
from hf_to_cc import H2O, integrals, scf_diis, ao2mo, spin_orbitals
from ccd_residual import residual
from ccd import ccd

mol, S, h, eri, E_nuc, N = integrals(H2O, "sto-3g")
E_hf, eps, C, n_scf = scf_diis(S, h, eri, E_nuc, N)
mo, h_mo = ao2mo(C, eri, h)
hso, g = spin_orbitals(mo, h_mo)
E_ccd, n_iter = ccd(residual, np.diag(np.repeat(eps, 2)), g, N)
```

## License

MIT, see [LICENSE](LICENSE).
