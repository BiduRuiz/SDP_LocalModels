# SDP_LocalModels

Welcome to the SDP_LocalModels repository! Here, I've documented a part of the results I generated during my master's thesis. 📜

## support folder 📂

Inside the "support" folder, you'll find a collection of files and folders. 
Please note that it might be a bit chaotic since it contains a history of trial and error experiments. 
Feel free to explore it, and you might discover something useful for your own research or projects! 🕵️‍♂️💡


## python files 🐍

- **utils.py**: this file has all the functions I've defined. 🧮
- **SDP_script.py**: This is an example script that showcases the practical application of the functions I've developed.
  It certifies whether a qubit state is entangled using a PPT criterion.
  Afterward, it assesses if the qubit state is unsteerable or local using a semidefinite programming (SDP) approach. 
    maximize $q$
    subject to $\text{tr}_A((\Pi_{a|\hat{v}_x}\otimes\mathbb{I}_B)O_{AB}) = \sum_\lambda D_\lambda(a|x)\rho_\lambda$, $\forall a,x$
              $\rho_\lambda\geq0$, $\forall\lambda$
              $rO_{AB}+(1-r)\xi\otimes O_B = \rho_q$.
  If the value of "q" found in the SDP equals 1, it indicates that the state is unsteerable (local). 📊
  
For more detailed insights and comprehensive explanations, please check out my master's thesis, which I've uploaded to this repository! 📚

Feel free to explore, experiment, and make the most of the resources here. 
If you have any questions or need further assistance, don't hesitate to reach out. 
Enjoy your exploration! 🚀🔬🌟
