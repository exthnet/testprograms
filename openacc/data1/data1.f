c -*- fortran -*-
      program main

      implicit none
      integer i, n
      double precision, allocatable :: v(:), y(:)
      integer iargc, nargc
      external iargc
      character*100 tmpstr
      integer iter

      nargc=iargc()
      if(nargc.ne.0)then
         call getarg(1,tmpstr)
         read(tmpstr,*) n
      else
         n = 10
      endif
      write(*,*)"n = ",n

      allocate(v(n))
      allocate(y(n))
      do i=1, n
         v(i) = dble(i)
         y(i) = 1.0d0
      enddo

      write(*,'(A)',advance="NO")"initial:"
      do I=1,N
         write(*,'(1H F6.2)',advance="NO")v(I)
      enddo
      write(*,*)""

      do iter=1, 10
         write(*,*)"iter=",iter

!$acc enter data copyin(v) pcreate(y)
!$acc update device(y)
!$acc kernels present(v,y)
!$acc loop
      do i=1, n
         v(i) = v(i) + y(i)
      enddo
!$acc end kernels
!$acc exit data

!$acc enter data
!$acc kernels present(v)
!$acc loop
      do i=1, n
         v(i) = v(i) + y(i)
      enddo
!$acc end kernels
!$acc exit data copyout(v)

      do i=1, n
         y(i) = y(i) + 1.0d0
      enddo

      enddo

      write(*,'(A)',advance="NO")"result:"
      do I=1,N
         write(*,'(1H F6.2)',advance="NO")v(I)
      enddo
      write(*,*)""

      deallocate(v)

 9999 stop
      end program main

