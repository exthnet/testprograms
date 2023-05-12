      program hello
      include 'mpif.h'
      integer ierr,myproc,hostnm,size
      character*64 hostname
      integer r,p
      r = MPI_THREAD_FUNNELED
      p = -1
      call mpi_init(ierr)
!      call mpi_init_thread(r,p,ierr)
      write(*,*)r,p
!$omp parallel
!$omp master
      call mpi_comm_rank(MPI_COMM_WORLD, myproc, ierr)
      call mpi_comm_size(MPI_COMM_WORLD, size, ierr)
      ierr=hostnm(hostname)
      write(6,100) myproc,size,hostname
      call mpi_barrier(MPI_COMM_WORLD,ierr)
!$omp end master
      write(*,*)"fin"
!$omp end parallel
 100        format(1x,"Hello, parallel world ",i3,"/",i3," : host ",A64)
      call mpi_finalize(ierr)
      end
