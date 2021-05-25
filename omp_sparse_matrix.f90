PROGRAM OMP_COO_MULT_VECTOR
USE OMP_LIB
IMPLICIT NONE

REAL(8), DIMENSION(:), ALLOCATABLE	:: A_VAL
INTEGER(4), DIMENSION(:), ALLOCATABLE	:: A_ROW, A_COL
REAL(8), DIMENSION(:), ALLOCATABLE	:: X
REAL(8), DIMENSION(:), ALLOCATABLE	:: B_LOCAL, B_TOTAL, B_REF
INTEGER(4)				:: A_N, N, I, M
INTEGER(4)				:: THREAD_RANK, THREAD_SIZE
REAL(8)					:: TSTART, TEND, TIME

! Define interfaces for external functions
INTERFACE

	SUBROUTINE READ_DATA(A_VAL, A_ROW, A_COL, X, B_TOTAL, B_REF, A_N, N)
        ! Read sparse matrix A (coo format, so 3*A_N values), vector X (which consists of 
        ! N elements) and result of matrix-vector multiplication B_REF (which consists of 
        ! N elements) which is computed by scipy.
        ! Allocates resourses for matrix-vector multiplication B_TOTAL which will be
        ! computed by this program.
        !
        ! :param A_VAL:         An array of matrix A elements
        ! :param A_ROW:         An array of row coordinates
        ! :param A_COL:         An array of column coordinates
        ! :param X:             Vector
        ! :param B_TOTAL:       Vector for futher matrix-vector multiplication result
        !                       storage
        ! :param B_REF:         Matrix-vector multiplication result which is computed 
        !                       by scipy and read from input file               
        ! :param A_N:           Number of non-zero elements in sparse matrix A
        ! :param N:             Dimension size

		REAL(8), DIMENSION(:), ALLOCATABLE, INTENT(OUT)	   :: A_VAL, X
		REAL(8), DIMENSION(:), ALLOCATABLE, INTENT(OUT)    :: B_TOTAL, B_REF
		INTEGER(4), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: A_COL, A_ROW
		INTEGER(4), INTENT(OUT)				   :: A_N, N
	END SUBROUTINE READ_DATA

END INTERFACE

	! Read sparse matrix A, vector X, result vector B, number of
        ! non-zero elements in matrix A_N and dimension size N.
	! Initialize local thread storage
	CALL READ_DATA(A_VAL, A_ROW, A_COL, X, B_TOTAL, B_REF, A_N, N)
	ALLOCATE(B_LOCAL(N))
	B_LOCAL = [(0.0D0, I = 1, N)]
	
	TSTART = OMP_GET_WTIME()
	!$OMP PARALLEL PRIVATE(THREAD_RANK, I, B_LOCAL) & 
	!$OMP SHARED(A_VAL, A_ROW, A_COL, A_N,          &
	!$OMP X, B_TOTAL, THREAD_SIZE, N, M)
		THREAD_RANK = OMP_GET_THREAD_NUM()
		THREAD_SIZE = OMP_GET_NUM_THREADS()
		M = A_N / THREAD_SIZE
		
		! ASSIGN JOB TO EACH THREAD AND COMPUTE THE LOCAL MULTIPLICATION RESULT
		DO I = THREAD_RANK * M + 1, (THREAD_RANK + 1) * M, 1
			B_LOCAL(A_ROW(I)) = B_LOCAL(A_ROW(I)) + A_VAL(I) * X(A_COL(I))
		ENDDO
		! GATHER LOCAL RESULTS INTO ONE SHARED ARRAY
		!$OMP CRITICAL
			DO I = 1, N, 1
				B_TOTAL(I) = B_TOTAL(I) + B_LOCAL(I)
			ENDDO
		!$OMP END CRITICAL
	!$OMP END PARALLEL
	TEND = OMP_GET_WTIME()
	TIME = TEND - TSTART
	
	! DEALLOCATE RESOURSES, PRINT NUMERIC ERROR AND COMPUTATION TIME
	WRITE(*, *) "NUMERIC ERROR: ", MAXVAL(ABS(B_REF(:) - B_TOTAL(:)))
	WRITE(*, *) "COMPUTATION TIME: ", TIME, " S."
	DEALLOCATE(A_VAL, A_ROW, A_COL, X, B_TOTAL, B_REF, B_LOCAL)
END PROGRAM OMP_COO_MULT_VECTOR


SUBROUTINE READ_DATA(A_VAL, A_ROW, A_COL, X, B_TOTAL, B_REF, A_N, N)
! Read sparse matrix A (coo format, so 3*A_N values), vector X (which consists of 
! N elements) and result of matrix-vector multiplication B_REF (which consists of 
! N elements) which is computed by scipy.
! Allocates resourses for matrix-vector multiplication B_TOTAL which will be
! computed by this program.
!
! :param A_VAL:         An array of matrix A elements
! :param A_ROW:         An array of row coordinates
! :param A_COL:         An array of column coordinates
! :param X:             Vector
! :param B_TOTAL:       Vector for futher matrix-vector multiplication result
!                       storage
! :param B_REF:         Matrix-vector multiplication result which is computed 
!                       by scipy and read from input file               
! :param A_N:           Number of non-zero elements in sparse matrix A
! :param N:             Dimension size

IMPLICIT NONE
REAL(8), DIMENSION(:), ALLOCATABLE, INTENT(OUT)    :: A_VAL, X
REAL(8), DIMENSION(:), ALLOCATABLE, INTENT(OUT)    :: B_TOTAL, B_REF
INTEGER(4), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: A_COL, A_ROW
INTEGER(4), INTENT(OUT)				   :: A_N, N
INTEGER(4), PARAMETER				   :: ISTREAM=100
INTEGER(4)					   :: I

	! Open input stream for reading file data
	OPEN(ISTREAM, FILE="sparse_matrix4.txt")
	
	! Read number of non-zero elements A_N in sparse matrix A
	READ(ISTREAM, *) A_N
	! Allocates the resources and read the data in coo format
	ALLOCATE(A_ROW(A_N), A_COL(A_N), A_VAL(A_N))
	DO I = 1, A_N, 1
		READ(ISTREAM, *) A_ROW(I), A_COL(I), A_VAL(I)
		! Number of elements in fortran starts from 1 while
		! the number of elements in python starting from 0
		A_ROW(I) = A_ROW(I) + 1
		A_COL(I) = A_COL(I) + 1
	ENDDO
	
	! Read number of elements in vector
	READ(ISTREAM, *) N
	! Read vector elements one by one
	ALLOCATE(X(N))	
	DO I = 1, N, 1
		READ(ISTREAM, *) X(I)
	ENDDO

	! Read number of elements in matrix-vector
	! multiplication result
	READ(ISTREAM, *) N
	! Read matrix-vector multiplication result
	! and initialize the local one by zero vector
	ALLOCATE(B_TOTAL(N), B_REF(N))
	DO I = 1, N, 1
		READ(ISTREAM, *) B_REF(I)
		B_TOTAL(I) = 0.0D0
	ENDDO
	
	! Close input stream
	CLOSE(ISTREAM)


END SUBROUTINE READ_DATA

