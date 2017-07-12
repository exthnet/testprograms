program gemvtest
  real*8, allocatable :: mat(*,*), vec(*)
  allocate(mat(4,4))
  allocate(vec(4))
  mat(1,1) = 1.0d0
  mat(2,1) = 1.0d0
  mat(3,1) = 1.0d0
  mat(4,1) = 1.0d0
  mat(1,2) = 1.0d0
  mat(2,2) = 1.0d0
  mat(3,2) = 1.0d0
  mat(4,2) = 1.0d0
  mat(1,3) = 1.0d0
  mat(2,3) = 1.0d0
  mat(3,3) = 1.0d0
  mat(4,3) = 1.0d0
  mat(1,4) = 1.0d0
  mat(2,4) = 1.0d0
  mat(3,4) = 1.0d0
  mat(4,4) = 1.0d0
  vec(1) = 2.0d0
  vec(2) = 2.0d0
  vec(3) = 2.0d0
  vec(4) = 2.0d0
  write(*,*)mat
  write(*,*)vec

  write(*,*)vec
end program gemvtest

