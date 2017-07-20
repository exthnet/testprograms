subroutine func(a)
  implicit none
  integer :: i, j
  integer :: a(:,:)
  do i=1, 4
     do j=1, 4
        write(*,*)a(j,i)
     enddo
  enddo
end subroutine func

program test
  implicit none
  integer :: i, j
  integer, allocatable :: a(:,:)

  allocate(a(4,4))
  do i=1, 4
     do j=1, 4
        a(j,i) = i+j
     enddo
  enddo

  call func(a)

  deallocate(a)
end program test


