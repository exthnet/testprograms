! -*- f90 -*-
program test
  implicit none
  real(kind=8), dimension(:,:), allocatable :: A, B, C
  integer :: N, nargs, nlength, nstatus, i, j
  character*256 :: value

  nargs=command_argument_count()
  if(nargs>0)then
     call get_command_argument(1,value,nlength,nstatus)
     read(value,*) N
  else
     N = 10
  end if
  write(*,*)"N=",N
  ! メモリの確保
  allocate(A(N,N), B(N,N), C(N,N))
  ! 計算
  do i = 1, N
     do j = 1, N
        C(i,j) = A(i,j) + B(i,j)
     end do
  end do
  ! 計算結果の利用
  print *, C
  ! メモリの解放
  deallocate(A, B, C)
end program test
