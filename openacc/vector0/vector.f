c -*- fortran -*-
      program main

      implicit none
      integer i, n
      double precision, allocatable :: v(:)
      integer iargc, nargc
      external iargc
      character*100 tmpstr

      nargc=iargc()
      if(nargc.ne.0)then
         call getarg(1,tmpstr)
         read(tmpstr,*) n
      else
         n = 10
      endif
      write(*,*)"n = ",n

      allocate(v(n))
      do i=1, n
         v(i) = dble(i)
      enddo

      write(*,'(A)',advance="NO")"initial:"
      do I=1,N
         write(*,'(1H F6.2)',advance="NO")v(I)
      enddo
      write(*,*)""

!$acc kernels
      do i=1, n
         v(i) = v(i) * 2.0d0
      enddo
!$acc end kernels

      write(*,'(A)',advance="NO")"result:"
      do I=1,N
         write(*,'(1H F6.2)',advance="NO")v(I)
      enddo
      write(*,*)""

      deallocate(v)

 9999 stop
      end program main

