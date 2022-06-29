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

The files are: SDP_Completo.py which has all the funcions already inside and it is way more clean, AND IS WORKING! For more commented and documented code, view the Jupyter Notebook LocalModels.ipynb. And LHV_alpha.py which is not working yet, just ignore it for now...

## Dependencies
- picos
- numpy
- math
- matplotlib
