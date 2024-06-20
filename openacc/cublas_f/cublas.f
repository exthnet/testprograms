! -*- f90 -*-

module gpukernel
contains
  attributes(global) subroutine gpukernel(N, C, A, B)
    integer,value :: ID, N
    real(kind=4), device, dimension(N), intent(in) :: A, B
    real(kind=4), device, dimension(N), intent(inout) :: C
    ID = (blockIdx%x-1)*blockDim%x + threadIdx%x
    if(ID.le.N)then
       C(ID) = C(ID) + A(ID) * B(ID)
    endif
  end subroutine gpukernel
end module gpukernel


program main
  use cudafor
  use cublas
  implicit none

  real(4), allocatable, dimension(:) :: A, B, C
  real(4), allocatable, dimension(:),device :: dA, dB, dC
  integer :: I, N, X
  type(dim3) :: dimGrid, dimBlock

  X = 10
  N = 128
  allocate(A(N), B(N), C(N))
  allocate(dA(N), dB(N), dC(N))

  C = 0.0
  do I=1, N
     A(I) = real(I)/real(N)
  enddo
  B = 2.0

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

  dA = A;   dB = B;   dC = C

!  dimGrid = dim3(2,1,1)
!  dimBlock = dim3(64,1,1)
!  call gpukernel<<<dimGrid, dimBlock>>>(N, dC, dA, dB)
      call cublassgemm()

  C = dC

  write(*,*) "C (after)"
  do I=1, N
     if(mod(I,X).eq.0)then
        write(*,'(F8.4)') C(I)
     else
        write(*,'(F8.4)',advance='no') C(I)
     endif
  enddo
  write(*,*) ""

  deallocate(dA, dB, dC)
  deallocate(A, B, C)

end program main
