! -*- fortran -*-
program main
  use omp_lib
  implicit none
  integer :: i, j
  integer :: n = 10
  integer :: iter, maxiter=100
  double precision :: cond = 1.0e-16
  double precision :: t1, t2
  double precision :: alpha, beta
  double precision, allocatable :: r(:), p(:), Ax(:)
  double precision :: bnrm, dnrm, resid, rho, rho1, pq
  double precision, allocatable :: z(:), q(:), dd(:)
  double precision, allocatable :: A(:,:), b(:), x(:), xx(:)
  double precision :: d
1001 format (F8.2)
1002 format (F8.2)
1008 format (E12.3,E12.3)
1010 format (I0,E12.3,E12.3,E12.3)

  allocate(A(N,N), b(N), x(N), xx(N))
  allocate(r(N), p(N), Ax(N), z(N), q(N), dd(N))

  do i=1, N
     do j=1, N
        A(j,i) = 1.0d0
     enddo
  enddo
  do i=1, N
     A(i,i) = dble(N-1)
  enddo

  do i=1, N
     x(i) = 0.0d0
     call random_number(d)
     xx(i) = dble(N) * d
  enddo
  do i=1, N
     b(i) = 0.0d0
     do j=1, N
        b(i) = b(i) + A(j,i) * xx(j)
     enddo
  enddo
  do i=1, N
     x(i) = 0.0d0
  enddo
#if 1
  write(*,*)"A"
  do i=1, N
     do j=1, N
        write(*,1001,advance='no')A(j,i)
     enddo
     write(*,*)""
  enddo
  write(*,*)"x"
  do i=1, N
     write(*,1002)xx(i)
  enddo
  write(*,*)"b"
  do i=1, N
     write(*,1002)b(i)
  enddo
#endif
  ! initialize
  ! {x}=0.0
  x(:) = 0.0d0
  r(:) = 0.0d0
  z(:) = 0.0d0
  q(:) = 0.0d0
  p(:) = 0.0d0
  dd(:) = 0.0d0

  do i=1, N
     dd(i) = 1.0d0 / A(i,i)
  end do

  ! {r0} = {b} - [A]{xini}
  do i=1, N
     Ax(i) = 0.0d0
     do j=1, N
        Ax(i) = Ax(i) + A(j,i) * x(j)
     end do
  end do
  do i=1, N
     r(i) = b(i) - Ax(i)
  end do

  bnrm = 0.0d0
  do i=1, N
     bnrm = bnrm + b(i) * b(i)
  end do
  if(bnrm.eq.0.0d0)bnrm=1.0d0

  t1 = omp_get_wtime
!$acc data copyin(z(N), dd(N), r(N), p(N), q(N), A(N,N)) copy(x(N))
  do iter=1, maxiter
     write(*,'(A,I)',advance="no")"iter",iter

     ! {z} = [Minv]{r}
     !$acc kernels
     !$acc loop
     do i=1, N
        z(i) = dd(i) * r(i)
     end do
     !$acc end kernels

     ! {rho} = {r}{z}
     rho = 0.0d0
     !$acc kernels
     !$acc loop
     do i=1, N
        rho = rho + r(i) * z(i)
     end do
     !$acc end kernels

     ! {p} = {z} if iter=1
     ! beta = rho/rho1 otherwise
     if(iter.eq.1) then
     !$acc kernels
     !$acc loop
        do i=1, N
           p(i) = z(i)
        end do
     !$acc end kernels
     else
        beta = rho / rho1
     !$acc kernels
     !$acc loop
        do i=1, N
           p(i) = z(i) + beta * p(i)
        end do
     !$acc end kernels
     end if

     ! {q} = [A]{p}
     !$acc kernels
     !$acc loop
     do i=1, N
        q(i) = 0.0
        do j=1, N
           q(i) = q(i) + A(j,i) * p(j)
        end do
     end do
     !$acc end kernels

     ! alpha = rho / {p}{q}
     pq = 0.0d0
     !$acc kernels
     !$acc loop
     do i=1, N
        pq = pq + p(i) * q(i)
     end do
     !$acc end kernels
     alpha = rho / pq

     ! {x} = {x} + alpha*{p}
     ! {r} = {r} - alpha*{q}
     !$acc kernels
     !$acc loop
     do i=1, N
        x(i) = x(i) + alpha * p(i)
        r(i) = r(i) - alpha * q(i)
     end do
     !$acc end kernels

     ! check converged
     dnrm = 0.0d0
     !$acc kernels
     !$acc loop
     do i=1, N
        dnrm = dnrm + r(i) * r(i)
     end do
     !$acc end kernels
     resid = sqrt(dnrm/bnrm)
     write(*,1008) resid,cond
     if(resid.le.cond)exit
     if(iter.eq.maxiter)exit

     rho1 = rho
  end do
!$acc end data
  t2 = omp_get_wtime();

  write(*,*)"time",t2-t1,"sec, ",(t2-t1)/iter,"sec/iter"
#if 1
  write(*,*)"x"
  do i=1, N
     write(*,1002)x(i)
  enddo
  write(*,*)"result"
  do i=1,N
     write(*,1010)i,xx(i),x(i),xx(i)-x(i)
  end do
#endif
  deallocate(dd)
  deallocate(q)
  deallocate(z)
  deallocate(Ax)
  deallocate(p)
  deallocate(r)

  deallocate(xx)
  deallocate(x)
  deallocate(b)
  deallocate(A)

9999 stop
end program

