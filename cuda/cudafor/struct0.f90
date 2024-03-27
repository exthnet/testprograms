! -*- f90 -*-
program stream
  implicit none

  type :: st_x
     integer :: value
     ! この動的メンバがある状態でmanaged向けコンパイルをすると問題が起きる
     integer,pointer :: tmp(:) => null()
  end type st_x

  integer   comm, ierr
  integer :: n, i
  integer :: ret
  integer(int_ptr_kind()) :: nFree, nTotal
  integer :: devID, tid, istat, ndevices
  integer, allocatable :: h_data(:)
  type (st_x), allocatable :: d_data(:)
  integer :: nargs, len, nlength, nstatus
  character*64 :: str

  ! 配列の長さを実行時引数で指定
  nargs=command_argument_count()
  if(nargs==0)then
     len = 10
  else
     call get_command_argument(1,str,nlength,nstatus)
     read(str,*)len
  endif
  write(*,*)"len=",len

  ! 配列の動的確保
  allocate(h_data(len))
  allocate(d_data(len))

  ! 値の設定
  do i=1, len
     h_data(i) = i
  end do

  ! 値の確認
  write(*,*)"before"
  write(*,*)h_data

  ! コピー
  d_data%value = h_data

  ! 計算
  do i=1, len
     d_data(i)%value = d_data(i)%value * 2
  end do

  ! 書き戻し
  h_data = d_data%value

  ! 結果の表示
  write(*,*)"after"
  write(*,*)h_data

  deallocate(d_data)
  deallocate(h_data)
end program stream

