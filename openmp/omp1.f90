module omp1
contains
  subroutine func2(a, n, fid)
    implicit none
    integer :: i, j, n, x
    integer, pointer :: a(:,:)
    integer :: fid
    write(fid,*)"func2"
    do i=1, n
       do j=1, n
          write(fid,'(i)',advance='no')a(j,i)
       enddo
       write(fid,*)''
    enddo
  end subroutine func2

  subroutine func1(a)
    use omp_lib
    implicit none
    integer :: i, j, x
    integer, pointer :: a(:,:)
    integer, pointer :: p(:,:)=>null()
    integer :: tid, fid
    tid = omp_get_thread_num()
    fid = 100 + tid
    if(tid.eq.0)open(fid,file="0.log")
    if(tid.eq.1)open(fid,file="1.log")
!$omp barrier
    if(tid.le.1)then
       write(fid,*)"func1"
       do i=1, 4
          do j=1, 4
             write(fid,'(i)',advance='no')a(j,i)
          enddo
          write(fid,*)''
       enddo

       call func2(a,4,fid)
       p => a
       call func2(p,3,fid)
       p => a(1+tid:4,1+tid:4)
!$omp barrier
       do x=1, 10
          call func2(p,3,fid)
       enddo
    endif
!$omp barrier
    close(fid)

  end subroutine func1
end module omp1

program test
  use omp1
  implicit none
  integer :: i, j
  integer, pointer :: a(:,:)

  allocate(a(4,4))
  do i=1, 4
     do j=1, 4
        a(j,i) = i+j
     enddo
  enddo

  write(*,*)"initial"
  do i=1, 4
     do j=1, 4
        write(*,'(i)',advance='no')a(j,i)
     enddo
     write(*,*)''
  enddo

!$omp parallel
  call func1(a)
!$omp end parallel

  deallocate(a)
end program test


