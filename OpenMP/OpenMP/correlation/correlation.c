#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1000. */
//#define MINI_DATASET
//#define SMALL_DATASET
//#define STANDARD_DATASET
#define LARGE_DATASET
//#define EXTRALARGE_DATASET
#include "correlation.h"

#include <omp.h>

/* Array initialization. */
static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,M,N,m,n))
{
  int i, j;

  *float_n = 1.2;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      data[i][j] = ((DATA_TYPE) i*j) / M;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))

{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_correlation(int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,M,N,m,n),
			DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{

  //omp_set_num_threads(8);

  int i, j, j1, j2;

  DATA_TYPE eps = 0.1f;

#define sqrt_of_array_cell(x,j) sqrt(x[j])

  /* Determine mean of column vectors of input data matrix */
    #pragma omp parallel for private(i) shared(mean, data, float_n)
    for (j = 0; j < _PB_M; j++)
      {
        mean[j] = 0.0;
	for (i = 0; i < _PB_N; i++)
	  mean[j] += data[i][j];
	mean[j] /= float_n;
      }

  /* Determine standard deviations of column vectors of data matrix. */
    #pragma omp parallel for private(i) shared(stddev, data, mean, float_n)
    for (j = 0; j < _PB_M; j++)
      {
        stddev[j] = 0.0;
	for (i = 0; i < _PB_N; i++)
	  stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
	stddev[j] /= float_n;
	stddev[j] = sqrt_of_array_cell(stddev, j);
	/* The following in an inelegant but usual way to handle
	   near-zero std. dev. values, which below would cause a zero-
	   divide. */
	stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
      }
    
  /* Center and reduce the column vectors. */
    #pragma omp parallel for private(j) shared(data, mean, stddev, float_n)
    for (i = 0; i < _PB_N; i++)
      for (j = 0; j < _PB_M; j++)
	{
          data[i][j] -= mean[j];
          data[i][j] /= sqrt(float_n) * stddev[j];
	}

      #pragma omp parallel for private(j2, i) shared(symmat, data) schedule(dynamic)
          for (j1 = 0; j1 < _PB_M-1; j1++) {
            symmat[j1][j1] = 1.0;
            for (j2 = j1+1; j2 < _PB_M; j2++) {
                double sum = 0.0; // Variabile privata per ogni thread
                for (i = 0; i < _PB_N; i++)
                    sum += (data[i][j1] * data[i][j2]);
                #pragma omp critical // Evita race condition durante l'aggiornamento di symmat
                {
                   symmat[j1][j2] = sum;
                   symmat[j2][j1] = sum;
                }
            }
        }
        symmat[_PB_M-1][_PB_M-1] = 1.0;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);

  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));
  
  /* Start timer. */
  polybench_start_instruments;
  /* Run kernel. */
  kernel_correlation (m, n, float_n,
		      POLYBENCH_ARRAY(data),
		      POLYBENCH_ARRAY(symmat),
		      POLYBENCH_ARRAY(mean),
		      POLYBENCH_ARRAY(stddev));
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;
  
  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat)));
  //print_array(m, POLYBENCH_ARRAY(symmat));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(symmat);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}
