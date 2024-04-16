program cublas_test
  use omp_lib
  use cublas
  implicit none
  real*8, allocatable, dimension(:,:) :: a, b, c
  integer :: i, j, k, n
  integer :: seedsize
  integer,allocatable :: seed(:)
  integer :: ret, nargs, nlength, nstatus
  character*256 value
  real*8 :: tmp
  real*8 :: t1, t2
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

  t1 = omp_get_wtime()
  !$acc host_data use_device(a,b,c)
  call cublasDgemm('N','N', n, n, n, 1.0d0, a, n, b, n, 0.0d0, c, n)
  !$acc end host_data
  !$acc wait
  t2 = omp_get_wtime()
  write(*,'(a,g12.4,a)')"time ",t2-t1," sec"

  write(*,*)"C"
  do j=1, n
     do i=1, n
        write(*,1000,advance='no')c(j,i)
     end do
     write(*,*)""
  end do


  deallocate(a, b, c)

end program cublas_test

