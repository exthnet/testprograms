! -*- fortran -*-
program main

  implicit none
  integer :: i, j, n=10
  double precision :: v1(10)

  do i=1, n
     v1(i) = dble(i)
  enddo

!$acc data copy(v1)
!$acc kernels
  do i=1, n
     v1(i) = v1(i) * 2.0d0
  enddo
!$acc end kernels

!$acc update host(v1)
  do i=1,n
     write(*,'(1H F8.2)',advance="NO")v1(i)
  enddo
  write(*,*)""

!$acc kernels
  do i=1, n
     v1(i) = v1(i) * 2.0d0
  enddo
!$acc end kernels
!$acc end data

  do i=1,n
     write(*,'(1H F8.2)',advance="NO")v1(i)
  enddo
  write(*,*)""

end program main
