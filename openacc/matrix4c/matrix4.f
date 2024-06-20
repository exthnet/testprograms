c -*- fortran -*-
      program main

      implicit none
      integer i, j, l
!      integer m, n, k
      integer m ! row of A & C
      integer n ! col of B & C
      integer k ! col of A, row of B
      double precision, allocatable :: a(:,:), b(:,:), c(:,:)
      integer iargc, nargc
      external iargc
      character*100 tmpstr
      integer out

      out = 1

      nargc=iargc()
      if(nargc.lt.4)then
         write(*,*)"usage: progn m n k (out)"
         goto 9999
      else
         call getarg(1,tmpstr)
         read(tmpstr,*) m
         call getarg(2,tmpstr)
         read(tmpstr,*) n
         call getarg(3,tmpstr)
         read(tmpstr,*) k
         if(nargc.ge.4)then
            call getarg(4,tmpstr)
            read(tmpstr,*) out
         endif
      endif
      write(*,*)"m,n,k = ",m,n,k

      allocate(a(k,m))
      allocate(b(n,k))
      allocate(c(n,m))
      do i=1, m
         do j=1, k
            a(j,i) = dble(i) + dble(j)/100.0d0
        enddo
      enddo
      do i=1, k
         do j=1, n
            b(j,i) = dble(i) + dble(j)/100.0d0
        enddo
      enddo
      do i=1, m
         do j=1, n
            c(j,i) = 0.0d0
        enddo
      enddo

!$acc kernels
!$acc loop independent gang,vector(32)
      do i=1, m
         do j=1, n
            do l=1, k
               c(j,i) = c(j,i) + a(l,i) * b(j,l)
            enddo
         enddo
      enddo
!$acc end kernels

      if(out.ne.0)then
      write(*,'(A)')"result:"
      do i=1,m
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

