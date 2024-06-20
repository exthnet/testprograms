! -*- fortran -*-
subroutine swap(a,n,m)
  integer :: a(*)
  integer :: n, m
  integer :: tmp
  tmp = a(n)
  a(n) = a(m)
  a(m) = tmp
end subroutine

program main
  implicit none
  integer :: i, j, n=10, x, y
  double precision, allocatable :: v1(:), v2(:)
  integer,allocatable :: index(:)
  double precision :: d
  integer iargc, nargc
  external iargc
  character*100 tmpstr

  nargc=iargc()
  if(nargc.ne.0)then
     call getarg(1,tmpstr)
     read(tmpstr,*) n
  else
     n = 10
  endif
  write(*,*)"n = ",n

  allocate(v1(n), v2(n), index(n))
  do i=1, n
     v1(i) = dble(i)
     v2(i) = 0.0d0
     index(i) = i
  enddo
  do i=1, n
     call random_number(d)
     x = int(d*dble(n)) + 1
     call random_number(d)
     y = int(d*dble(n)) + 1
     call swap(index,x,y)
  enddo
  write(*,*) index

!$acc kernels
  do i=1, n
     v2(i) = v1(index(i)) * 2.0d0
  enddo
!$acc end kernels

  do i=1,n
     write(*,'(1H F8.2)',advance="NO")v2(i)
  enddo
  write(*,*)""

end program main
