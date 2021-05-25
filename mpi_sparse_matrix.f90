PROGRAM MPI_COO_MULT_VECTOR
IMPLICIT NONE
INCLUDE 'mpif.h'

INTEGER(4), PARAMETER			:: MASTER_RANK=0
REAL(8), DIMENSION(:), ALLOCATABLE	:: A_VAL, A_SUB_VAL
INTEGER(4), DIMENSION(:), ALLOCATABLE	:: A_ROW, A_COL, A_SUB_ROW, A_SUB_COL
REAL(8), DIMENSION(:), ALLOCATABLE	:: X
REAL(8), DIMENSION(:), ALLOCATABLE	:: B_LOCAL, B_TOTAL, B_REF
INTEGER(4)				:: A_N, N, M, I
INTEGER(4)				:: PROC_RANK, PROC_SIZE, IERROR
REAL(8)					:: TSTART, TEND, LOCAL_TIME, TOTAL_TIME

! Define interfaces for external functions
INTERFACE

	SUBROUTINE READ_DATA(A_VAL, A_ROW, A_COL, X, B_TOTAL, B_REF, A_N, N)
	! Read sparse matrix A (coo format, so 3*A_N values), vector X (which consists of 
	! N elements) and result of matrix-vector multiplication B_REF (which consists of 
	! N elements) which is computed by scipy.
	! Allocates resourses for matrix-vector multiplication B_TOTAL which will be
	! computed by this program.
	!
	! :param A_VAL: 	An array of matrix A elements
	! :param A_ROW: 	An array of row coordinates
	! :param A_COL:		An array of column coordinates
	! :param X:		Vector
	! :param B_TOTAL:	Vector for futher matrix-vector multiplication result
	!			storage
	! :param B_REF:		Matrix-vector multiplication result which is computed 
	!			by scipy and read from input file		
	! :param A_N:		Number of non-zero elements in sparse matrix A
	! :param N:		Dimension size
		REAL(8), DIMENSION(:), ALLOCATABLE, INTENT(OUT)      :: A_VAL, X
		REAL(8), DIMENSION(:), ALLOCATABLE, INTENT(OUT)      :: B_TOTAL, B_REF
		INTEGER(4), DIMENSION(:), ALLOCATABLE, INTENT(OUT)   :: A_COL, A_ROW
		INTEGER(4), INTENT(OUT)                              :: A_N, N
	END SUBROUTINE READ_DATA
	
END INTERFACE

	CALL MPI_INIT(IERROR)
	
	!Initializing matrix, vector and additional data for parallel computation
	CALL MPI_COMM_RANK(MPI_COMM_WORLD, PROC_RANK, IERROR)	
	CALL MPI_COMM_SIZE(MPI_COMM_WORLD, PROC_SIZE, IERROR)
	IF (PROC_RANK == MASTER_RANK) THEN
		! Read sparse matrix A, vector X, result vector B, number of
 		! non-zero elements in matrix A_N and dimension size N
		CALL READ_DATA(A_VAL, A_ROW, A_COL, X, B_TOTAL, B_REF, A_N, N)
		M = A_N / PROC_SIZE
		!WRITE(*, *) "A_ROW", A_ROW
		!WRITE(*, *) "A_COL", A_COL
		!WRITE(*, *) "A_VAL", A_VAL
	ENDIF
	CALL MPI_BCAST(N, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD, IERROR)	
	CALL MPI_BCAST(M, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD, IERROR)
	IF (PROC_RANK /= MASTER_RANK) THEN
		ALLOCATE(X(N))
	ENDIF
	CALL MPI_BCAST(X, N, MPI_DOUBLE_PRECISION, MASTER_RANK, MPI_COMM_WORLD, IERROR)
	ALLOCATE(B_LOCAL(N))
	B_LOCAL = [(0.0D0, I = 1, N)]
	ALLOCATE(A_SUB_VAL(M), A_SUB_ROW(M), A_SUB_COL(M))
	
	! ASSIGN JOB TO EACH PROCCESS AND COMPUTE THE MULTIPLICATION RESULT
	TSTART = MPI_WTIME(IERROR)
	CALL MPI_SCATTER(A_VAL, M, MPI_DOUBLE_PRECISION, &
			 A_SUB_VAL, M, MPI_DOUBLE_PRECISION, &
			 MASTER_RANK, MPI_COMM_WORLD, IERROR)
	CALL MPI_SCATTER(A_ROW, M, MPI_INT, &
			 A_SUB_ROW, M, MPI_INT, &
			 MASTER_RANK, MPI_COMM_WORLD, IERROR)
	CALL MPI_SCATTER(A_COL, M, MPI_INT, &
			 A_SUB_COL, M, MPI_INT, &
			 MASTER_RANK, MPI_COMM_WORLD, IERROR)
	DO I = 1, M, 1
		B_LOCAL(A_SUB_ROW(I)) = B_LOCAL(A_SUB_ROW(I)) + A_SUB_VAL(I) * X(A_SUB_COL(I))
	ENDDO	
	CALL MPI_REDUCE(B_LOCAL, B_TOTAL, N, MPI_DOUBLE_PRECISION, &
			MPI_SUM, MASTER_RANK, MPI_COMM_WORLD, IERROR)	
	TEND = MPI_WTIME(IERROR)
	LOCAL_TIME = TEND - TSTART
	!WRITE(*, *) "B_LOCAL: ", B_LOCAL
	CALL MPI_REDUCE(LOCAL_TIME, TOTAL_TIME, 1, MPI_DOUBLE_PRECISION, &
			MPI_MAX, MASTER_RANK, MPI_COMM_WORLD, IERROR)

	! DEALLOCATE RESOURSES. PRINT NUMERIC ERROR AND COMPUTATION TIME
	DEALLOCATE(A_SUB_VAL, A_SUB_ROW, A_SUB_COL)
	DEALLOCATE(B_LOCAL, X)
	IF (PROC_RANK == MASTER_RANK) THEN
		WRITE(*, *) "NUMERIC ERROR: ", MAXVAL(ABS(B_REF(:) - B_TOTAL(:)))
		WRITE(*, *) "COMPUTATION TIME: ", TOTAL_TIME, " S."
		DEALLOCATE(A_VAL, A_ROW, A_COL, B_TOTAL, B_REF)
	ENDIF
	
	CALL MPI_FINALIZE(IERROR)

END PROGRAM MPI_COO_MULT_VECTOR


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
REAL(8), DIMENSION(:), ALLOCATABLE, INTENT(OUT)	:: A_VAL, X
REAL(8), DIMENSION(:), ALLOCATABLE, INTENT(OUT)	:: B_TOTAL, B_REF
INTEGER(4), DIMENSION(:), ALLOCATABLE, INTENT(OUT)	:: A_COL, A_ROW
INTEGER(4), INTENT(OUT)					:: A_N, N
INTEGER(4), PARAMETER					:: ISTREAM=100
INTEGER(4)						:: I


	! Open input stream for reading file data
	OPEN(ISTREAM, FILE="sparse_matrix4.txt")

        ! Read number of non-zero elements A_N in sparse matrix A
	READ(ISTREAM, *) A_N
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
	
	CLOSE(ISTREAM)


END SUBROUTINE READ_DATA
