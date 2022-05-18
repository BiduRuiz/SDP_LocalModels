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

We are starting with 3 measurements, each with 2 results.

The files are: LHS_alpha.py (calls the functions), WernerClass.py (creates the class of states), estrategias.py (creates the deterministic strategies), poly.py (creates and plots the polyhedrons and measurements), polyhedron.py (is not used anymore, but it has the polyhedrons without keeping the measurements fixed) and SDP_alpha.py (the SDP function). If instead of find \alpha (with is q in the SDP) you just want to find the local model, then you can fix the q in SDP_model.py and run LHS_model.py.

I also created SDP_Completo.py which has all the funcions already inside and it is way more clean...

LHV_alpha calls a diferent SDP, which is in SDP_alpha, but it is not working, just ignore it for now...
## Dependencies
- picos
- numpy
- math
- matplotlib
