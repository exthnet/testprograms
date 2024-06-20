program alloc
  implicit none
  real*8, allocatable :: d(:,:,:)
  integer :: ig
  allocate(d(,,10))
  do i=1,10
     allocate(d(
  end do
  deallocate(d)
end program alloc

