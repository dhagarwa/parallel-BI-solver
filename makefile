include ${PETSC_DIR}conf/base
output: potential.cc
	${CLINKER} potential.cc -o output ${PETSC_LIB}

clean:
	rm *.o output
