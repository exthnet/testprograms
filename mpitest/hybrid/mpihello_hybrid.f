      program hello
      implicit none
      include 'mpif.h'
      integer ierr,myproc,hostnm
      character*64 hostname
      integer tid, tall
      integer omp_get_thread_num, omp_get_num_threads

      call mpi_init(ierr)
      call mpi_comm_rank(MPI_COMM_WORLD, myproc, ierr)
      ierr=hostnm(hostname)
!$omp parallel
      tid = omp_get_thread_num()
      tall = omp_get_num_threads()
      write(6,100) myproc,hostname,tid,tall
!$omp end parallel
 100    format(1x,"hello - I am process",i3," host ",A64,":",i3,"/",i3)
      call mpi_finalize(ierr)
      end
