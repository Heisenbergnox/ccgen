# ccgen-wick — generating coupled-cluster equations from second quantization

Coupled-cluster equations get long quickly: the CCSD equations below come out as 328 terms. ccgen derives them the way you would on paper, contraction by contraction with Wick's theorem, and writes them out as NumPy code that you can run and check.

**Coming next:**
- merging like terms into the textbook form (see the known limitation);
- CCSDT and EOM-CC;
- plugging the generated equations into the solvers of [hf_to_cc](https://github.com/Heisenbergnox/hf_to_cc);
- faster generated code: shared intermediates instead of one `einsum` per term.

## What's inside

ccgen currently generates CCSD and CCD. With $`\bar H = e^{-T} H_N\, e^{T}`$, the CCSD equations (T = T₁ + T₂) are

```math
E_\text{CCSD} = \langle \Phi_0 | \bar H | \Phi_0 \rangle = \sum_{ia} f_{ia}\, t_i^a + \frac{1}{4} \sum_{ijab} \langle ij \Vert ab \rangle\, t_{ij}^{ab} + \frac{1}{2} \sum_{ijab} \langle ij \Vert ab \rangle\, t_i^a t_j^b ,
```

```math
0 = \langle \Phi_i^a | \bar H | \Phi_0 \rangle , \qquad 0 = \langle \Phi_{ij}^{ab} | \bar H | \Phi_0 \rangle .
```

CCD (T = T₂) has one amplitude equation, $`0 = \langle \Phi_{ij}^{ab} | \bar H | \Phi_0 \rangle`$, and its energy is the CCSD energy without the terms that contain t₁.

- `ccgen.py`: the generator. Each part of an equation, for example $`\langle \Phi_{ij}^{ab} | (V_N T_1 T_2)_c | \Phi_0 \rangle`$, is a list of normal-ordered operators built by `make_ops(bra, h, ts)`: `bra` is the excitation level of the projection (0 for the energy, 1 for the T₁ equation, 2 for the T₂ equation), `h` is `"f"` or `"v"`, and `ts` lists the cluster operators, for example `[1, 2]` for T₁T₂. For each list the generator
  1. finds the allowed, connected full contractions;
  2. takes each sign from the number of crossing lines;
  3. renames the summed indices and attaches integrals, amplitudes and the coefficient: 1/(n!)² for each operator with n creators, times 1/k! for k equal cluster operators, from expanding $`e^T`$;
  4. brings each term to a standard form and merges equal ones;
  5. writes one `einsum` line per term, with `optimize=True` so that NumPy contracts two arrays at a time.
- `provided.py`: lists the parts of each equation and writes the generated code (`python provided.py`):
  - `ccd_residual.py`: `residual(f, g, t2)`, 125 terms;
  - `ccsd_residual.py`: `Eccsd`, `Rccsdt1` and `Rccsdt2`, all taking `(f, g, t1, t2)`, with 3, 32 and 293 terms.
- `ccd.py`, `ccsd.py`: `ccd(R, f, g, N)` and `ccsd(f, g, N, Eccsd, Rccsdt1, Rccsdt2)`, which iterate the generated equations, starting from MP2 amplitudes and zero t₁.
- `run.py`: the example in Usage.

### What goes in

**Index letters** (spin orbitals): i, j, k, l, m, n are occupied; a, b, c, d, e, f are virtual; p, q, r, s run over all orbitals and are the Hamiltonian's indices before contraction. In a part, i, j, a, b belong to the projection and stay as the free indices of the result. The cluster operators use c–f and k–n, which are summed over.

**Input format.** An operator is a tuple `(index, "+" or "-", space, group)`. The space is `"o"` (occupied), `"v"` (virtual) or `"g"` (general); the group says which operator it belongs to: 0 for the projection, 1 for the Hamiltonian, 2, 3, … for the cluster operators. For example `make_ops(2, "v", [2])`, the part $`\langle \Phi_{ij}^{ab} | (V_N T_2)_c | \Phi_0 \rangle`$, is

```
[('i', '+', 'o', 0), ('j', '+', 'o', 0), ('b', '-', 'v', 0), ('a', '-', 'v', 0),
 ('p', '+', 'g', 1), ('q', '+', 'g', 1), ('s', '-', 'g', 1), ('r', '-', 'g', 1),
 ('c', '+', 'v', 2), ('d', '+', 'v', 2), ('l', '-', 'o', 2), ('k', '-', 'o', 2)]
```

Each part is one connected term of $`\bar H`$ projected onto the bra of its equation, named by its operators: VT₁T₂ is the $`(V_N T_1 T_2)_c`$ term, and VT₁² carries a factor ½ from expanding $`e^T`$. `provided.py` passes every combination of `"f"` or `"v"` with the cluster operators up to the order needed. Combinations without a connected full contraction give no terms, so nothing has to be picked out by hand. The parts that do give terms:

| Equation | Parts (number of terms) | Total |
|---|---|---|
| CCD T₂ | V 1, FT₂ 8, VT₂ 18, VT₂² 98 | 125 |
| CCSD energy | FT₁ 1, VT₂ 1, VT₁² 1 | 3 |
| CCSD T₁ | F 1, FT₁ 2, FT₂ 4, FT₁² 2, VT₁ 1, VT₂ 4, VT₁² 4, VT₁T₂ 8, VT₁³ 6 | 32 |
| CCSD T₂ | V 1, FT₂ 8, FT₁T₂ 8, VT₁ 4, VT₂ 18, VT₁² 12, VT₁T₂ 44, VT₂² 98, VT₁³ 24, VT₁²T₂ 52, VT₁⁴ 24 | 293 |

The generated CCSD energy, for example, is

```python
def Eccsd(f, g, t1, t2):
    N = t2.shape[0]
    o = slice(0, N)
    v = slice(N, None)
    r = 1.0 * np.einsum('kc,kc->', f[o, v], t1, optimize=True)
    r += 0.25 * np.einsum('klcd,klcd->', g[o, o, v, v], t2, optimize=True)
    r += 0.5 * np.einsum('klcd,kc,ld->', g[o, o, v, v], t1, t1, optimize=True)
    return r
```

which is the energy formula above, term by term.

## Results

Water in STO-3G: CCD converges in 29 iterations to −0.0506409699 Eh and CCSD in 33 iterations to −0.0509029526 Eh (correlation energies); together they take under a second.

- CCSD: for 14 molecules in STO-3G, from H₂ to CH₃OH, the correlation energy agrees with PySCF's CCSD (converged to 1e-12) to within 1e-9 Eh, for most of them to about 1e-11 Eh. The largest differences (CO 6e-10, CO₂ 1e-10) come from the SCF orbitals: on PySCF's own orbitals, with the full Fock matrix, CO agrees to 5e-12 Eh.
- CCSD in larger basis sets: water in 6-31G and cc-pVDZ also agrees with PySCF to about 1e-11 Eh; the cc-pVDZ run (24 basis functions) takes 9 s.
- CCD: on 18 molecule/basis runs, the result agrees with the hand-written CCD of hf_to_cc to 1e-9 Eh.

## Known limitation

Like terms are not merged yet. Terms that are the same up to the names of the summed indices stay separate, and terms related by swapping i↔j or a↔b are not yet collected into the P(ij), P(ab) of the textbook form. So the CCD residual comes out as 125 `einsum` lines instead of the textbook's ten terms, and CCSD as 3 + 32 + 293. The results are correct, only longer and slower than they need to be; merging like terms is next on the list.

## Usage

The example takes its integrals from [hf_to_cc](https://github.com/Heisenbergnox/hf_to_cc), so clone both repositories next to each other and install the dependencies:

```
git clone https://github.com/Heisenbergnox/hf_to_cc
git clone https://github.com/Heisenbergnox/ccgen-wick
pip install numpy scipy pyscf
cd ccgen-wick
```

The generated files `ccd_residual.py` and `ccsd_residual.py` are already in the repository. To generate them again, for example after changing the generator or the lists of parts, run

```
python provided.py
```

in the `ccgen-wick` directory. It needs nothing beyond the Python standard library, takes about 0.1 s, and overwrites both files.

Then run `PYTHONPATH=../hf_to_cc python run.py`. It prints the CCD and the CCSD correlation energy (Eh) with the number of iterations: −0.0506409699 and 29, −0.0509029526 and 33.

```python
import numpy as np
from hf_to_cc import H2O, integrals, scf_diis, ao2mo, spin_orbitals
from ccd_residual import residual
from ccd import ccd
from ccsd_residual import Eccsd, Rccsdt1, Rccsdt2
from ccsd import ccsd
mol, S, h, eri, E_nuc, N = integrals(H2O, "sto-3g")
E_hf, eps, C, n_scf = scf_diis(S, h, eri, E_nuc, N)
mo, h_mo = ao2mo(C, eri, h)
hso, g = spin_orbitals(mo, h_mo)
E_ccd, n_iter_ccd = ccd(residual, np.diag(np.repeat(eps, 2)), g, N)
E_ccsd, n_iter_ccsd = ccsd( np.diag(np.repeat(eps, 2)), g, N, Eccsd, Rccsdt1, Rccsdt2)
print(E_ccd, n_iter_ccd)
print(E_ccsd, n_iter_ccsd)
```

The arrays use spin orbitals with the `N` occupied ones first: `g[p, q, r, s]` is $`\langle pq \Vert rs \rangle`$, `t1[i, a]` is $`t_i^a`$ and `t2[i, j, a, b]` is $`t_{ij}^{ab}`$. `f` is the Fock matrix in the same basis. For canonical Hartree–Fock orbitals it is diagonal, with the orbital energies on the diagonal, which is what `np.diag(np.repeat(eps, 2))` builds; the generated equations also take a full Fock matrix, including its occupied–virtual block.

## License

MIT, see [LICENSE](LICENSE).
