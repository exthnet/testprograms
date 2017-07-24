module mod
contains
  subroutine func1(X,Y)
    implicit none
    external :: omp_get_thread_num, omp_get_num_threads
    integer*4 :: omp_get_thread_num, omp_get_num_threads
    integer :: tid
    real*8, pointer :: X(:)
    real*8, pointer :: Y(:)
    real*8, pointer :: p(:)
    p => null()

    tid = omp_get_thread_num()
    write(*,*)"tid",tid
    !$omp barrier
    if(tid.eq.0)then
       p => X
    endif
    !$omp barrier
    if(tid.eq.1)then
       p => Y
    endif
    !$omp barrier

    !$omp critical
    write(*,*)"func1", tid, p(1)
    !$omp end critical

  end subroutine func1

  subroutine func2(X,Y)
    implicit none
    external :: omp_get_thread_num, omp_get_num_threads
    integer*4 :: omp_get_thread_num, omp_get_num_threads
    integer :: tid
    real*8, pointer :: X(:)
    real*8, pointer :: Y(:)
    real*8, pointer :: p(:)=>null()

    tid = omp_get_thread_num()
    write(*,*)"tid",tid
    !$omp barrier
    if(tid.eq.0)then
       p => X
    endif
    !$omp barrier
    if(tid.eq.1)then
       p => Y
    endif
    !$omp barrier

    !$omp critical
    write(*,*)"func2", tid, p(1)
    !$omp end critical

  end subroutine func2
end module mod

program fomp2
  use mod
  implicit none
  external :: omp_get_thread_num, omp_get_num_threads
  integer*4 :: omp_get_thread_num, omp_get_num_threads
  real*8, pointer :: X(:) => null()
  real*8, pointer :: Y(:) => null()
  integer :: i, N
  N = 10

  allocate(X(N))
  allocate(Y(N))
  do i=1,N
     X(i) = 1.0d0
     Y(i) = 2.0d0
  enddo

!$omp parallel
  call func1(X,Y)
!$omp end parallel

!$omp parallel
  call func2(X,Y)
!$omp end parallel

  deallocate(X)
  deallocate(Y)
end program fomp2
