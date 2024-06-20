c -*- fortran -*-

      subroutine sub(N, N2)

      IMPLICIT NONE
      INTEGER I, N, N2
!      PARAMETER (N=10)
      DOUBLE PRECISION sum
!      DOUBLE PRECISION D(N)
      real(kind=8), dimension(N+N2):: D

!$acc enter data
!$acc& create(D)

!$acc kernels present(D)
      DO I=1,N
         D(I) = dble(I)
      ENDDO
!$acc end kernels

!$acc kernels present(D)
      DO I=1,N
         D(I) = D(I) * 2.0d0
      ENDDO
      sum = 0.0d0
      DO I=1,N
         sum = sum + D(I)
      ENDDO
!$acc end kernels

!      write(*,'(A)',advance="NO")"result:"
!      DO I=1,N
!         write(*,'(1H F6.2)',advance="NO")D(I)
!      ENDDO
!      write(*,*)""

!$acc exit data

      write(*,*)"sum=",sum
      end subroutine


      PROGRAM MAIN

      call sub(4,6)

 9999 STOP
      END PROGRAM

