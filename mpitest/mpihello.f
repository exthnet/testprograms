      program hello
      include 'mpif.h'
      integer ierr,myproc,hostnm
      character*64 hostname
      call mpi_init(ierr)
      call mpi_comm_rank(MPI_COMM_WORLD, myproc, ierr)
!      ierr=setvbuf3f(6,2,0)
      ierr=hostnm(hostname)
      write(6,100) myproc,hostname
 100    format(1x,"hello - I am process",i3," host ",A64)
      call mpi_finalize(ierr)
      end
