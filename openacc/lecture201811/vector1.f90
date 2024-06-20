! -*- fortran -*-
program main

  implicit none
  integer :: i, n=10
  double precision :: v1(10), v2(10)

  do i=1, n
     v1(i) = dble(i)
     v2(i) = 0.0d0
  enddo

!$acc kernels copyout(v2(:)) copyin(v1(:))
!$acc loop gang, vector(32)
  do i=1, n
     v2(i) = v1(i) * 2.0d0
  enddo
!$acc end kernels

  do i=1,n
     write(*,'(1H F8.2)',advance="NO")v1(i)
  enddo
  write(*,*)""
  do i=1,n
     write(*,'(1H F8.2)',advance="NO")v2(i)
  enddo
  write(*,*)""

end program main
