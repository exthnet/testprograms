program cublas_test
  use cublas
  implicit none
  real*8, allocatable, dimension(:,:) :: a, b, c
  real*8, allocatable, dimension(:,:), device :: da, db, dc
  integer :: i, j, k, n
  integer :: seedsize
  integer,allocatable :: seed(:)
  integer :: ret, nargs, nlength, nstatus
  character*256 value
  real*8 :: tmp
1000 format (1x,f6.2)

  nargs=command_argument_count()
  if(nargs.ne.1)then
     write(*,*)"usage: ./a.out n"
     stop
  endif
  call get_command_argument(1,value,nlength,nstatus)
  if(value.ne."")then
     read(value,*) n
  endif
  write(*,*)"n=",n

  allocate(a(n,n))
  allocate(b(n,n))
  allocate(c(n,n))
  allocate(da(n,n))
  allocate(db(n,n))
  allocate(dc(n,n))

  call random_seed(size=seedsize)
  allocate(seed(seedsize))
  seed = 1
  call random_seed(put=seed)

  do i=1, n
     do j=1, n
        !call random_number(tmp)
        !a(i,j) = ceiling(tmp*10.0d0)
        a(i,j) = real(i*2+j)
     end do
  end do
  do i=1, n
     do j=1, n
        !call random_number(tmp)
        !b(i,j) = ceiling(tmp*10.0d0)
        b(i,j) = real(i*2+j)
     end do
  end do
  do i=1, n
     do j=1, n
        c(i,j) = 0.0d0
     end do
  end do

  write(*,*)"A"
  do j=1, n
     do i=1, n
        write(*,1000,advance='no')a(j,i)
     end do
     write(*,*)""
  end do

  write(*,*)"B"
  do j=1, n
     do i=1, n
        write(*,1000,advance='no')b(j,i)
     end do
     write(*,*)""
  end do

  da = a
  db = b
  dc = c
  !call cublas_init
  call cublasDgemm('N','N', n, n, n, 1.0d0, da, n, db, n, 0.0d0, dc, n)
  !call cublas_shutdown
  c = dc

  write(*,*)"C"
  do j=1, n
     do i=1, n
        write(*,1000,advance='no')c(j,i)
     end do
     write(*,*)""
  end do


  deallocate(a, b, c)
  deallocate(da, db, dc)

end program cublas_test

