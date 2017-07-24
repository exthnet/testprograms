program fomp2
  implicit none
  external :: omp_get_thread_num, omp_get_num_threads
  integer*4 :: omp_get_thread_num, omp_get_num_threads
  real*8, pointer :: X(:) => null()
  real*8, pointer :: Y(:) => null()
  real*8, pointer :: p(:) => null()
  integer :: i, tid
  integer :: N
  N = 10

  allocate(p(N))
  allocate(X(N))
  allocate(Y(N))
  do i=1,N
     X(i) = 1.0d0
     Y(i) = 2.0d0
  enddo

!$omp parallel private(tid,p)
  tid = omp_get_thread_num()
  if(tid.eq.0)then
     p => X
  endif
!$omp barrier
  if(tid.eq.1)then
     p => Y
  endif
!$omp barrier
!$omp critical
  write(*,*)tid, p(1)
!$omp end critical
!$omp end parallel

  deallocate(X)
  deallocate(Y)
end program fomp2
