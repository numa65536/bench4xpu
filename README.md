# bench4xpu

L'objectif de ces codes est multiple :
- tester efficacement une xPU (CPU ou GPU) 
- explorer son empreinte de parallélisme
- charger le composant pour évaluer sa consommation électrique
- charger le composant pour évaluer sa réponse climatique
- servir de "code matrice" pour d'autres projets

Classiquement, quand je teste une nouvelles xPU, j'exploite les 3 suivants :
- **NBody.py** : calculs flottants à grain fin, empreinte mémoire faible, échanges mémoire importants ;
- **Pi** : calculs flottants et entiers (majoritaires) à gros grain, empreinte mémoire faible, échanges mémoire faibles ;
- **ETSN** : calculs flottants à gros grain, empreinte mémoire importante, échange mémoire importants.

Dans ce dossier **bench4xpu**, vous avez plusieurs dossiers :
- **BLAS** contenant les dossiers :
  - **xGEMM** : implémentations multiples de multiplication matrice-matrice
  - **xTRSV** : implémentations multiples d'un enchaînement de fonction BLAS
* **Epidevomath** : un prototype d'implémentation sur GPU d'un projet (abandonné)
- **FFT** contenant une première exploitation de cuFFT (en suspens)
- **Ising** : implémentations multiples du modèle d'Ising en Python (multiples parallélisations)
- **NBody** : implémentation en PyOpenCL d'un modèle N-Corps newtonien
- **Pi** : implémentation multiple d'un Pi Monte Carlo
- **Splutter** : implémentation en PyOpenCL d'un postillonneur mémoire
- **TrouNoir** : un exemple de portage de code de 1994, porté en C en 1997 puis en Python/OpenCL et Python/CUDA en 2019
- **ETSN** : les programmes corrigés de l'école d'été ETSN 2022

Les environnements de programmation CPU et GPU explorés dans ces exemples varient d'un programme à l'autre :

- **BLAS** : multiplication matrice-matrice, résolution de système triangulaire
  - en C : FBLAS, CBLAS, CuBLAS, CuBLAS Thunking, CLBLAS,
- **NBody** : PyOpenCL avec PyOpenGL
- **Pi** : programme gros grain, majoritairement calcul entier
  - Python : PyOpenCL, PyCUDA, Python/MPI/OpenCL, PythoN
  - C : MPI, OpenMP, MPI/OpenMP, OpenACC, SyCL, Kokkos
- **TrouNoir** : Fortran77, 
  - Fortran77 : l'originel de 1996 avec PGplot
  - C : C, C/OpenMP, C/OpenACC, C/SyCL
  - Python : PyCUDA, PyOpenCL
