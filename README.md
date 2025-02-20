[![Powered by Simone](https://img.shields.io/badge/Author%20-Simone%20Colli-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)]()  
[![Contacts](https://img.shields.io/badge/Email%20-simone.colli.96@gmail.com-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)]()  
[![Contacts](https://img.shields.io/badge/Email%20-218735@studenti.unimore.it@gmail.com-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)]()  

--- 

# HPC---A-parallelization-of-a-Correlation-problem
Il repo contiene 3 implementazioni: sequenziale, OpenMP e CUDA del calcolo della matrice di correlazione.

---

## Obiettivi
Ottimizzare (parallelizzare) l'esecuzione delle applicazioni assegnate su un sistema multiprocessore.

• Analizzare il codice  
• Individuare i punti critici adatti alla parallelizzazione  
• Utilizzare strumenti di profiling  
• Usare il modello di programmazione OpenMP & CUDA  
• Comprendere le prestazioni ottenute

---
## Utilizzo
Per eseguire il la versione sequenziale:
```bash
cd ./OpenMP/sequential/correlation
make clean
make run
```
Per eseguire il la versione OpenMP:
```bash
cd ./OpenMP/OpenMP/correlation
make clean
make run
```
Per eseguire il la versione CUDA:
```bash
cd ./OpenMP/CUDA/correlation
make clean
./run.sh
```

---

## Profiling & Benchmarking
• Configurazioni
    - controllare il readme.md ./OpenMp
---

### Polybench
Per sfruttare il calcolo del tempo di esecuzione:
• aprire il file ***./HPC---A-parallelization-of-a-Correlation-problem/OpenMP/utilities/common.mk***  
• inserire ***-DPOLYBENCH_TIME*** nell'istruzione ***OPT=-O3 -g -fopenmp***

Automaticamente a fine esecuzione verrà mostrato il tempo di esecuzione.
### Gprof
Per fare profiling con ***Gprof***:  
• aprire il file ***./HPC---A-parallelization-of-a-Correlation-problem/OpenMP/utilities/common.mk***  
• inserire ***-pg*** nell'istruzione ***OPT=-O3 -g -fopenmp***

Verrà generato un file ***gmon.out*** sul quale lanciando il comando:
```bash
  gprof correlation_acc gmon.out > output.txt.
```
si genererà un file ***output.txt*** con il risultato dell'esecuzione.
### Valgrind
Per utilizzare ***Valgrind***:
```bash
valgrind --tool=callgrind ./correlation_acc
callgrind_annotate callgrind.out.* oppure kcachegrind callgrind.out.<pid>
```


---
