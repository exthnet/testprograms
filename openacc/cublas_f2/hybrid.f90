! -*- f90 -*-

program main
  use cublas
  use cudafor
  implicit none

  real(4), allocatable, dimension(:,:) :: A, B, C
  real(4) :: alpha, beta
  integer :: N, I, J, K
  integer :: t1, t2, t_rate, t_max, diff
  type(cublasHandle) :: handle
  integer(kind=cuda_stream_kind) :: stream

  character*100 TMPSTR
  INTEGER iargc, nargc
  EXTERNAL iargc
  nargc=iargc()
  if(nargc.eq.0)then
     write(*,*)"usage: ./execfile N"
     stop
  else
     call getarg(1,TMPSTR)
     read(TMPSTR,*)N
     write(*,*)"N=",N
  endif

  allocate(A(N,N), B(N,N), C(N,N))

  C = 0.0
  do I=1, N
     do J=1, N
        A(J,I) = real(I-1) + real(J-1)/10.0
        B(J,I) = real(I-1) + real(J-1)/10.0
     enddo
  enddo
  alpha = 1.0;  beta = 1.0

!$acc enter data copyin(A(1:N,1:N), B(1:N,1:N), C(1:N,1:N))
  I = cublasCreate(handle)
  J = cublasGetStream(handle,stream)
  K = cudaStreamSynchronize(stream)
  call system_clock(t1)
!$acc host_data use_device(A, B, C)
  call cublassgemm('n','n',N,N,N,alpha,A,N,B,N,beta,C,N)
!$acc end host_data
  J = cublasGetStream(handle,stream)
  K = cudaStreamSynchronize(stream)
  call system_clock(t2, t_rate, t_max)
  I = cublasDestroy(handle)
!$acc exit data copyout(C(1:N,1:N))

  if(t2.lt.t1)then
     diff = (t_max - t1) + t2 + 1
  else
     diff = t2 - t1
  endif
  print "(A, F20.8)", "TIME:", diff/dble(t_rate)

  deallocate(A, B, C)
end program main
