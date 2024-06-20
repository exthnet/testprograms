c -*- fortran -*-
      program main

      implicit none
      integer i, j, k, n
      double precision, allocatable :: a(:,:), b(:,:), c(:,:)
      integer iargc, nargc
      external iargc
      character*100 tmpstr
      integer out

      out = 1

      nargc=iargc()
      if(nargc.eq.0)then
         n = 10
      else
         call getarg(1,tmpstr)
         read(tmpstr,*) n
         if(nargc.gt.1)then
            out = 0
         endif
      endif
      write(*,*)"n = ",n

      allocate(a(n,n))
      allocate(b(n,n))
      allocate(c(n,n))
      do i=1, n
         do j=1, n
            a(j,i) = dble(i) + dble(j)/100.0d0
            b(j,i) = dble(i) + dble(j)/100.0d0
            c(j,i) = 0.0d0
        enddo
      enddo

!$acc kernels
!$acc loop independent gang
      do i=1, n
!$acc loop independent vector
         do j=1, n
!$acc loop seq
            do k=1, n
               c(j,i) = c(j,i) + a(k,i) * b(j,k)
            enddo
         enddo
      enddo
!$acc end kernels

      if(out.eq.1)then
      write(*,'(A)')"result:"
      do i=1,n
         do j=1,n
            write(*,'(1H F6.2)',advance="NO")c(j,i)
         enddo
         write(*,'(A)')""
      enddo
      write(*,*)""
      endif

      deallocate(a)
      deallocate(b)
      deallocate(c)

 9999 stop
      end program main

