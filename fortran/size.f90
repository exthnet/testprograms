program size
  implicit none
  include 'mpif.h'
  integer :: i
  integer :: j(MPI_STATUS_SIZE)
  integer :: len
  !write(*,*)"size(i) = ", size(i)
  !write(*,*)"size(j) = ", size(j)
  inquire (iolength = len) i
  write(*,*)"size(i) = ", len
  inquire (iolength = len) j
  write(*,*)"size(j) = ", len
end program size
