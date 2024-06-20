program main
  use gpukernel
  implicit none

!  real(kind=4), allocatable, dimension(:) :: A, B, C
  real(4), allocatable, dimension(:) :: A, B, C
  integer :: I, N, X

  X = 10
  N = 128
  allocate(A(N), B(N), C(N))

  C = 0.0;  B = 2.0
  do I=1, N
     A(I) = real(I)/real(N)
  enddo

  write(*,*) "A"
  do I=1, N
     if(mod(I,X).eq.0)then
        write(*,'(F8.4)') A(I)
     else
        write(*,'(F8.4)',advance='no') A(I)
     endif
  enddo
  write(*,*) ""
  write(*,*) "B"
  do I=1, N
     if(mod(I,X).eq.0)then
        write(*,'(F8.4)') B(I)
     else
        write(*,'(F8.4)',advance='no') B(I)
     endif
  enddo
  write(*,*) ""
  write(*,*) "C (before)"
  do I=1, N
     if(mod(I,X).eq.0)then
        write(*,'(F8.4)') C(I)
     else
        write(*,'(F8.4)',advance='no') C(I)
     endif
  enddo
  write(*,*) ""


!$acc enter data copyin(A(1:N), B(1:N), C(1:N))

!$acc kernels
!$acc loop
  do I=1, N
     C(I) = C(I) + A(I) * B(I)
  enddo
!$acc end kernels

!$acc host_data use_device(A, B, C)
  call gpukernel_wrapper(N, C, A, B)
!$acc end host_data

!$acc exit data copyout(C(1:N))

  write(*,*) "C (after)"
  do I=1, N
     if(mod(I,X).eq.0)then
        write(*,'(F8.4)') C(I)
     else
        write(*,'(F8.4)',advance='no') C(I)
     endif
  enddo
  write(*,*) ""

  deallocate(A, B, C)

end program main
