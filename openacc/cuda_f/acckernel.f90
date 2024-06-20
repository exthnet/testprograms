module acckernel
contains
  subroutine acckernel(N, C, A, B)
!    integer,value :: I, N
!    real(kind=4), device, dimension(N), intent(in) :: A, B
!    real(kind=4), device, dimension(N), intent(inout) :: C
    integer :: I, N
    real(4), device :: A(:), B(:), C(:)
!$acc kernels deviceptr(A,B,C)
!$acc loop
    do I=1, N
       C(I) = C(I) + A(I) * B(I)
    enddo
    !$acc end kernels
  end subroutine acckernel
end module acckernel
