! -*- f90 -*-
program mm
  implicit none
  include 'mpif.h'
  real*8, allocatable :: a(:,:,:), b(:,:,:), c(:,:,:)
  integer :: i, j, k
  integer :: n, slice, iter
  integer :: nargs, nlength, nstatus, rank
  character*256 :: tmpstr
  real*8 :: tmpsum
  integer :: ierr
  real*8 :: t_begin, t_end, t_comm, t_comm1, t_comm2, t_calc, t_calc1, t_calc2
  integer :: seedsize
  integer, allocatable :: seed(:)

  slice = 10
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  write(*,*)"rank=",rank
  nargs=command_argument_count()
  if(nargs==0) then
     n = 10
  else
     call get_command_argument(1,tmpstr,nlength,nstatus)
     read(tmpstr,*)n
  end if
  write(*,*) "n=",n
  allocate(a(n,n,slice))
  allocate(b(n,n,slice))
  allocate(c(n,n,slice))

  call random_seed(size=seedsize)
  allocate(seed(seedsize))
  seed = 1
  call random_seed(put=seed)
  if(rank==0) then
     do iter=1,slice
        do i=1, n
           do j=1, n
              a(j,i,iter) = frand(1.0d0) !dble(i)/dble(n) + dble(j)/dble(n)/10.0d0
           end do
        end do
        do i=1, n
           do j=1, n
              b(j,i,iter) = frand(1.0d0) !dble(i)/dble(n) + dble(j)/dble(n)/10.0d0
           end do
        end do
        do i=1, n
           do j=1, n
              c(j,i,iter) = 0.0d0
           end do
        end do
     end do
  else
     a = 0.0d0
     b = 0.0d0
     c = 0.0d0
  endif

  t_comm = 0.0d0
  t_calc = 0.0d0
  t_begin = MPI_Wtime()
  do iter=1,slice
     t_comm1 = MPI_Wtime()
     call MPI_Bcast(a(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
     call MPI_Bcast(b(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
     call MPI_Bcast(c(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
     call MPI_Bcast(a(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
     call MPI_Bcast(b(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
     call MPI_Bcast(c(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
     call MPI_Bcast(a(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
     call MPI_Bcast(b(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
     call MPI_Bcast(c(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
     t_comm2 = MPI_Wtime()
     t_comm = t_comm + (t_comm2-t_comm1)
     t_calc1 = MPI_Wtime()
     !$omp parallel do collapse(2) private(k)
     do i=1, n
        do j=1, n
           do k=1, n
              c(j,i,iter) = c(j,i,iter) + a(j,k,iter) * b(k,i,iter)
           end do
        end do
     end do
     t_calc2 = MPI_Wtime()
     t_calc = t_calc + (t_calc2-t_calc1)
  end do
  t_end = MPI_Wtime()

  tmpsum = 0.0d0
  do iter=1,slice
     do i=1, n
        do j=1, n
           tmpsum = tmpsum + c(j,i,slice)
        end do
     end do
  end do

  write(*,fmt='(i0,1x,a,1pe15.7)') rank, "result: sum=", tmpsum
  write(*,fmt='(i0,1x,a,1pe15.7,1x,1pe15.7,1x,1pe15.7)') rank, "result: time=", t_end - t_begin, t_comm, t_calc

  deallocate(a)
  deallocate(b)
  deallocate(c)

  call MPI_Finalize(ierr)
contains
  real*8 function frand(d)
    implicit none
    real*8 :: x, d
    call random_number(x)
    frand = x * d
  end function frand
end program mm
