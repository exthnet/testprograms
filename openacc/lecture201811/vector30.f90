! -*- fortran -*-

module mod
contains

subroutine calc(n,v)
!$acc routine
  integer :: n
  double precision :: v(*)
!$acc loop
  do i=1, n
     v(i) = v(i) * 2.0d0
  enddo
end subroutine calc

end module mod

program main
  use mod
  implicit none
  integer :: i, j, n=10
  double precision :: v1(10)

  do i=1, n
     v1(i) = dble(i)
  enddo

!$acc data copy(v1)
!$acc kernels
  call calc(n,v1)
!$acc end kernels
!$acc end data

  do i=1,n
     write(*,'(1H F8.2)',advance="NO")v1(i)
  enddo
  write(*,*)""

end program main
