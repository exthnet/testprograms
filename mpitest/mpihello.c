#include <stdio.h>
#include <mpi.h>

int main (int argc, char **argv)
{
  int rank, size;
  int name_len;
  char name[MPI_MAX_PROCESSOR_NAME];
  
  MPI_Init (&argc, &argv);/* starts MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);/* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);/* get number of processes */
  MPI_Get_processor_name(name, &name_len);
  printf( "Hello, parallel world %d / %d : %s\n", rank, size, name ); fflush(stdout);
  MPI_Finalize();
  return 0;
}
