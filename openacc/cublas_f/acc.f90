! -*- f90 -*-

program main
  use cublas
  implicit none

  real(4), allocatable, dimension(:,:) :: A, B, C
  real(4) :: alpha, beta
  integer :: N, I, J, K

  N = 8
  allocate(A(N,N), B(N,N), C(N,N))

  C = 0.0
  do I=1, N
     do J=1, N
        A(J,I) = real(I-1) + real(J-1)/10.0
        B(J,I) = real(I-1) + real(J-1)/10.0
     enddo
  enddo
  alpha = 1.0;  beta = 1.0

  write(*,*) "A"
  do I=1, N
     do J=1, N
        write(*,'(F8.3)',advance='no') A(J,I)
     enddo
     write(*,*)""
  enddo
  write(*,*) "B"
  do I=1, N
     do J=1, N
        write(*,'(F8.3)',advance='no') B(J,I)
     enddo
     write(*,*)""
  enddo
  write(*,*) "C (before)"
  do I=1, N
     do J=1, N
        write(*,'(F8.3)',advance='no') C(J,I)
     enddo
     write(*,*)""
  enddo

!$acc enter data copyin(A(1:N,1:N), B(1:N,1:N), C(1:N,1:N))
!$acc kernels
!$acc loop collapse(2)
  do I=1, N
     do J=1, N
!$acc loop seq
        do K=1, N
           C(J,I) = C(J,I) + A(K,I) * B(J,K)
        enddo
     enddo
  enddo
!$acc end kernels
!$acc exit data copyout(C(1:N,1:N))

  write(*,*) "C (after)"
  do I=1, N
     do J=1, N
        write(*,'(F8.3)',advance='no') C(J,I)
     enddo
     write(*,*)""
  enddo

  deallocate(A, B, C)
end program main
