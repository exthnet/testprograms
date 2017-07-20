subroutine func(a)
  implicit none
  integer :: a(*)
  write(*,*)a(1)
end subroutine func

program test
  implicit none
  integer :: i
  integer, allocatable :: a(:)

  allocate(a(10))
  do i=1, 10
     a(i) = i
  enddo

  call func(a)
  call func(a(1))
  call func(a(2))

  deallocate(a)
end program test


