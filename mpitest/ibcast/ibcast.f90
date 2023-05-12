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
  integer, allocatable :: req(:), status(:)

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

  allocate(req(slice*9))
  allocate(status(MPI_STATUS_SIZE))

  t_comm = 0.0d0
  t_calc = 0.0d0
  t_begin = MPI_Wtime()

  t_comm1 = MPI_Wtime()
  iter=1
  call MPI_IBcast(a(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(1), ierr)
  call MPI_IBcast(b(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(2), ierr)
  call MPI_IBcast(c(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(3), ierr)
  call MPI_IBcast(a(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(4), ierr)
  call MPI_IBcast(b(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(5), ierr)
  call MPI_IBcast(c(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(6), ierr)
  call MPI_IBcast(a(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(7), ierr)
  call MPI_IBcast(b(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(8), ierr)
  call MPI_IBcast(c(1,1,iter), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(9), ierr)
  t_comm2 = MPI_Wtime()
  t_comm = t_comm + (t_comm2-t_comm1)

  do iter=1,slice
     t_comm1 = MPI_Wtime()
     if(iter<slice)then
        call MPI_IBcast(a(1,1,iter+1), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(iter*9+1), ierr)
        call MPI_IBcast(b(1,1,iter+1), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(iter*9+2), ierr)
        call MPI_IBcast(c(1,1,iter+1), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(iter*9+3), ierr)
        call MPI_IBcast(a(1,1,iter+1), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(iter*9+4), ierr)
        call MPI_IBcast(b(1,1,iter+1), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(iter*9+5), ierr)
        call MPI_IBcast(c(1,1,iter+1), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(iter*9+6), ierr)
        call MPI_IBcast(a(1,1,iter+1), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(iter*9+7), ierr)
        call MPI_IBcast(b(1,1,iter+1), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(iter*9+8), ierr)
        call MPI_IBcast(c(1,1,iter+1), n*n, MPI_REAL8, 0, MPI_COMM_WORLD, req(iter*9+9), ierr)
     endif
     call MPI_Wait(req((iter-1)*9+1), status, ierr)
     call MPI_Wait(req((iter-1)*9+2), status, ierr)
     call MPI_Wait(req((iter-1)*9+3), status, ierr)
     call MPI_Wait(req((iter-1)*9+4), status, ierr)
     call MPI_Wait(req((iter-1)*9+5), status, ierr)
     call MPI_Wait(req((iter-1)*9+6), status, ierr)
     call MPI_Wait(req((iter-1)*9+7), status, ierr)
     call MPI_Wait(req((iter-1)*9+8), status, ierr)
     call MPI_Wait(req((iter-1)*9+9), status, ierr)
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

  deallocate(req)
  deallocate(status)

  tmpsum = 0.0d0
  do iter=1,slice
     do i=1, n
        do j=1, n
           tmpsum = tmpsum + c(j,i,iter)
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
