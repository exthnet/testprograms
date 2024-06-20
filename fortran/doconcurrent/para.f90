program test
  implicit none
  real*8, allocatable, dimension(:,:) :: a, b, c
  integer :: i, j, k
  integer :: n, nargs, nlength, nstatus
  character*256 :: tmpstr

  nargs = command_argument_count()
  if(nargs==0)then
     n = 10
  else
     call get_command_argument(1, tmpstr, nlength, nstatus)
     read(tmpstr,*) n
  endif
  print *, "n=", n
  allocate(a(n,n),b(n,n),c(n,n))

  do i=1, n
     do j=1, n
        a(j,i) = 1.0d0
        b(j,i) = 1.0d0
        c(j,i) = 0.0d0
  end do
  end do

  write(*,*) "a"
  do i=1, n
     do j=1, n
        write(*,fmt='(1pe10.3)',advance='no') a(j,i)
     end do
     write(*,*)""
  end do
  write(*,*) "b"
  do i=1, n
     do j=1, n
        write(*,fmt='(1pe10.3)',advance='no') b(j,i)
     end do
     write(*,*)""
  end do
  write(*,*) "c"
  do i=1, n
     do j=1, n
        write(*,fmt='(1pe10.3)',advance='no') c(j,i)
     end do
     write(*,*)""
  end do

  do concurrent (i=1:n, j=1:n)
        c(j,i) = a(j,i) + b(j,i)
  end do

  write(*,*) "c"
  do i=1, n
     do j=1, n
        write(*,fmt='(1pe10.3)',advance='no') c(j,i)
     end do
     write(*,*)""
  end do

  deallocate(c, b, a)
end program test
