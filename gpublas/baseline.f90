program cublas_test
  use omp_lib
  implicit none
  real*8, allocatable, dimension(:,:) :: a, b, c
  integer :: i, j, k, n, out
  integer :: seedsize
  integer,allocatable :: seed(:)
  integer :: ret, nargs, nlength, nstatus
  character*256 value
  real*8 :: tmp
  real*8 :: t1, t2
1000 format (1x,f6.2)

  out = 1
  nargs=command_argument_count()
  if(nargs.lt.1)then
     write(*,*)"usage: ./a.out n (out)"
     stop
  endif
  call get_command_argument(1,value,nlength,nstatus)
  if(value.ne."")then
     read(value,*) n
  endif
  write(*,'(a,i0)')"n = ",n
  if(nargs.ge.2)then
     call get_command_argument(2,value,nlength,nstatus)
     if(value.ne."")then
        read(value,*) out
     endif
  endif

  call random_seed(size=seedsize)
  allocate(seed(seedsize))
  seed = 1
  call random_seed(put=seed)

  allocate(a(n,n))
  allocate(b(n,n))
  allocate(c(n,n))

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

  if(out.eq.1)then
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
  endif

  t1 = omp_get_wtime()
  do i=1, n
     do j=1, n
        do k=1, n
           c(i,j) = c(i,j) + a(i,k) * b(k,j)
        end do
     end do
  end do
  t2 = omp_get_wtime()
  write(*,'(a,g12.4,a)')"time ",t2-t1," sec"

  if(out.eq.1)then
     write(*,*)"C"
     do j=1, n
        do i=1, n
           write(*,1000,advance='no')c(j,i)
        end do
        write(*,*)""
     end do
  endif

  deallocate(a)
  deallocate(b)
  deallocate(c)

end program cublas_test

