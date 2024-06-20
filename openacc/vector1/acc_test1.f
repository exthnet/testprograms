c -*- fortran -*-
      PROGRAM MAIN

      IMPLICIT NONE
      INTEGER I, N
      PARAMETER (N=10)
      DOUBLE PRECISION D(N)

!$acc kernels loop copyin(D[1:N]) copyout(D[1:N])
      DO I=1,N
         D(I) = dble(I)
      ENDDO
!$acc end loop

      write(*,'(A)',advance="NO")"result:"
      DO I=1,N
         write(*,'(1H F6.2)',advance="NO")D(I)
      ENDDO
      write(*,*)""

 9999 STOP
      END PROGRAM

