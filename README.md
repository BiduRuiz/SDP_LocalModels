# SDP_LocalModels

## LHS - 2qubits
The SDP that we are working with is
<pre><code>maximize q
subject to
  tr_A(A_{a|x}\otimes\mathbb{1}\chi) = \sum_\lambda D_\lambda (a|x)\sigma_\lambda \forall a,x
  \eta\chi+(1-\eta)\xi\otimes\chi_B = q\rho+(1-q)\rho_{sep}
  \sigma_\lambda\geq0 \forall\lambda
</code></pre>
The target state is the class of two-qubit Werner states

\rho_W(\alpha)=\alpha\ket{\psi^-}\bra{\psi^-}+(1-\alpha)\mathbb{1}/4.

We are starting with 6 measurements, each with 2 results.

The files are: LHSqubit.py (calls the functions), estrategias.py (creates the deterministic strategies), polyhedron.py (creates and plots the polyhedrons) and SDP_LHS.py (the SDP function).
  
## Dependencies
- picos
- numpy
- math
- matplotlib
