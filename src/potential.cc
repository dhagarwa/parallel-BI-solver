#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <cstdlib>
#include <petscksp.h>
#include <petscmat.h>
using namespace std;
typedef  double T;

struct Sargs
{
    int npoints, ntri;
    T *points;
    int *tri;
    
};


void readTriangulation(int *tri, T *points, int ntri, int npoints )
{

    ifstream inFile;
    
    inFile.open("triangles5.txt");
    if (!inFile) {
        cout << "Unable to open file";
        exit(1); // terminate with error
    }
    
    for(int i=0; i < ntri; i++)
    {

    	inFile>>tri[3*i]>>tri[3*i+1]>>tri[3*i+2];	
    }

    inFile.close();

    inFile.open("points5.txt");
    
    if (!inFile) {
        cout << "Unable to open file";
        exit(1); // terminate with error
    }
   
    for(int i=0; i < npoints; i++)
    {

	inFile>>points[3*i]>>points[3*i+1]>>points[3*i+2];
    }
    
    inFile.close();
}


//Function to check if vertex with given vertex id is in traingle with given tr id
int inTriangle(int *tri, int tri_id, int vtx_id)
{
	if(tri[3*tri_id+0] == vtx_id)
		return 1;
	else if (tri[3*tri_id+1] == vtx_id)
		return 2;
	else if (tri[3*tri_id+2] == vtx_id)
		return 3;
	else 
		return 0;
	//return (tri[3*tri_id+0] == vtx_id || tri[3*tri_id+1] == vtx_id || tri[3*tri_id+2] == vtx_id );
}


//Function to calculate double layer kernel
void DLKernel(T *trg_point, T *src_tri, int n_src, T *K)
{
	T radius=1.0;
	T nrm; 
	T r1[3]; 
	for(int i=0; i < n_src; i++)
	{
		r1[0] = src_tri[3*i+0];
		r1[1] = src_tri[3*i+1];
		r1[2] = src_tri[3*i+2];
		nrm = sqrt((src_tri[3*i+0] - trg_point[0])*(src_tri[3*i+0] - trg_point[0]) +(src_tri[3*i+1] - trg_point[1])*(src_tri[3*i+1] - trg_point[1]) + (src_tri[3*i+2] - trg_point[2])*(src_tri[3*i+2] - trg_point[2]));
		
		if(nrm ==0) nrm = 1;
		K[i] = 1/(4*M_PI)*(1/nrm)*((trg_point[0] - r1[0])*r1[0]/radius + (trg_point[1] - r1[1])*r1[1]/radius + (trg_point[2] - r1[2])*r1[2]/radius )/(nrm*nrm);


	}
}



//Function to calculate double layer kernel while doing singular integration. Includes singular jacobian beforehand
void DLKernelSingular(T *trg_point, T *src_tri, int n_src, T *K, T *X)
{
	T radius=1.0;
	T nrm; 
	T r1[3];
        T X1[] = {0.212340538, 0.59053313, 0.91141204, 0.212340538, 0.59053313, 0.91141204, 0.212340538, 0.59053313, 0.91141204}; 
        if(X == NULL) X = X1; 
	for(int i=0; i < n_src; i++)
	{
		r1[0] = src_tri[3*i+0];
		r1[1] = src_tri[3*i+1];
		r1[2] = src_tri[3*i+2];
		nrm = sqrt((src_tri[3*i+0] - trg_point[0])*(src_tri[3*i+0] - trg_point[0]) +(src_tri[3*i+1] - trg_point[1])*(src_tri[3*i+1] - trg_point[1]) + (src_tri[3*i+2] - trg_point[2])*(src_tri[3*i+2] - trg_point[2]))/X[i];
		
		if(nrm ==0) nrm = 1;
		K[i] = 1/(4*M_PI)*(1/nrm)*((trg_point[0] - r1[0])*r1[0]/radius + (trg_point[1] - r1[1])*r1[1]/radius + (trg_point[2] - r1[2])*r1[2]/radius )/(nrm*nrm);


	}
}



//Function to tranform the gauss nodes on square to nodes on ref element using inverse Duffy transform 
void singularTransform(T *src_refx, T *src_refy, T *X, T *Y)
{

	int ngauss = 3; //order of gaussian quadrature
	T X1[] = {0.212340538, 0.59053313, 0.91141204, 0.212340538, 0.59053313, 0.91141204, 0.212340538, 0.59053313, 0.91141204}; 
        T Y1[] = {0.188409406, 0.523979067, 0.808694385, 0.106170269, 0.2952665677, 0.45570602024, 0.023931132, 0.0665540678, 0.1027176548}; //storing harcoded coordinates for gauss nodes on ref elements

	if(X == NULL ) X = X1;
	if(Y == NULL ) Y = Y1;
	T xi, eta;
	for(int i=0; i < 9; i++)
	{
		xi = X[i]; eta = Y[i];
		src_refx[i] = xi;
		src_refy[i] = xi*eta; 
	}

}

//Function for standardize Transform
void standardizeTransform(int *tri, int tri_id, T *points, T *src_tri, int circshift, T *X, T *Y)
{
	int temp;
	int ngauss = 3; //order of gaussian quadrature
	T X1[] = {0.212340538, 0.59053313, 0.91141204, 0.212340538, 0.59053313, 0.91141204, 0.212340538, 0.59053313, 0.91141204}; 
        T Y1[] = {0.188409406, 0.523979067, 0.808694385, 0.106170269, 0.2952665677, 0.45570602024, 0.023931132, 0.0665540678, 0.1027176548}; //storing harcoded coordinates for gauss nodes on ref elements

	if(X == NULL) X = X1;
	if(Y == NULL) Y = Y1;
	T xi, eta;
	T vtx1[3], vtx2[3], vtx3[3]; //three vertices of triangle in consideration
	int vtx1_id, vtx2_id, vtx3_id;
	vtx1_id = tri[3*tri_id + 0] - 1;
	vtx2_id = tri[3*tri_id + 1] - 1;
	vtx3_id = tri[3*tri_id + 2] - 1;
	if(circshift == 2) 
	{
		temp = vtx1_id;
		vtx1_id = vtx2_id;
		vtx2_id = vtx3_id;
		vtx3_id = temp;
	}
	else if(circshift == 3)
	{
		temp = vtx1_id;
		vtx1_id = vtx3_id;
		vtx3_id = vtx2_id;
		vtx2_id = temp;
	}
	vtx1[0] = points[3*vtx1_id+0]; vtx1[1] = points[3*vtx1_id+1]; vtx1[2] = points[3*vtx1_id+2];
	vtx2[0] = points[3*vtx2_id+0]; vtx2[1] = points[3*vtx2_id+1]; vtx2[2] = points[3*vtx2_id+2];
	vtx3[0] = points[3*vtx3_id+0]; vtx3[1] = points[3*vtx3_id+1]; vtx3[2] = points[3*vtx3_id+2];
	
	for(int i=0; i < 9; i++)
	{
		xi = X[i]; eta = Y[i];
		src_tri[3*i + 0] = (1-xi)*vtx1[0]  + (xi - eta)*vtx2[0] + (eta)*vtx3[0];
		src_tri[3*i + 1] =  (1-xi)*vtx1[1]  + (xi - eta)*vtx2[1] + (eta)*vtx3[1];

		src_tri[3*i + 2] =  (1-xi)*vtx1[2]  + (xi - eta)*vtx2[2] + (eta)*vtx3[2];

	}	

}


void zero(T *arr, int size)
{
	for(int i=0; i < size; i++)
	{
		arr[i] = 0.0;
	}
}

void crossprod(T *a, T *b, T *cross)
{

	cross[0] = a[1]*b[2] - a[2]*b[1];
	cross[1] = a[2]*b[0] - a[0]*b[2];
	cross[2] = a[0]*b[1] - a[1]*b[0];	
}


T norm(T *a)
{
	return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}


T gaussInt(T *K, T *den, int ngauss, T *Wx, T *Wy) //Gaussian quadrature over triangle
{
	T Wx1[] = {0.0698269799, 0.2292411063, 0.200931913738959}; //Default weights on reference element
	T Wy1[] = {0.27777777, 0.444444444, 0.277777777};
	T KWy[3];

	if(Wx == NULL) Wx = Wx1;
	if(Wy == NULL) Wy = Wy1;
	//for(int i=0; i < 9; i++) cout<<K[i]<<" ";
	KWy[0] = K[0]*den[0]*Wy[0] + K[3]*den[3]*Wy[1] + K[6]*den[6]*Wy[2]; 
	KWy[1] = K[1]*den[1]*Wy[0] + K[4]*den[4]*Wy[1] + K[7]*den[7]*Wy[2]; 
	KWy[2] = K[2]*den[2]*Wy[0] + K[5]*den[5]*Wy[1] + K[8]*den[8]*Wy[2]; 
	T val;
	val = Wx[0]*KWy[0] + Wx[1]*KWy[1] + Wx[2]*KWy[2];
	return val; 

}

T standardizeJac(int *tri, int tri_id, T *points )
{
	T cross[3];
	T a[3], b[3];
	T vtx1[3], vtx2[3], vtx3[3]; //three vertices of triangle in consideration
	int vtx1_id, vtx2_id, vtx3_id;
	vtx1_id = tri[3*tri_id + 0] - 1;
	vtx2_id = tri[3*tri_id + 1] - 1;
	vtx3_id = tri[3*tri_id + 2] - 1;
	vtx1[0] = points[3*vtx1_id+0]; vtx1[1] = points[3*vtx1_id+1]; vtx1[2] = points[3*vtx1_id+2];
	vtx2[0] = points[3*vtx2_id+0]; vtx2[1] = points[3*vtx2_id+1]; vtx2[2] = points[3*vtx2_id+2];
	vtx3[0] = points[3*vtx3_id+0]; vtx3[1] = points[3*vtx3_id+1]; vtx3[2] = points[3*vtx3_id+2];

	a[0] = vtx2[0] - vtx1[0];	
	a[1] = vtx2[1] - vtx1[1];	
	a[2] = vtx2[2] - vtx1[2];	
	b[0] = vtx3[0] - vtx1[0];	
	b[1] = vtx3[1] - vtx1[1];	
	b[2] = vtx3[2] - vtx1[2];	

	crossprod(a, b, cross);
	return norm(cross);
	


}

//Function to linear interpolate density to obtain values on gauss points in reference element
void interpolate(int *tri, int tri_id, T *eta, T *out, T *X, T *Y)
{
	T val, x, y, temp1, temp2, phi1, phi2, phi3;

	phi1 = eta[tri[tri_id*3+0]-1];
	phi2 = eta[tri[tri_id*3+1]-1];
	phi3 = eta[tri[tri_id*3+2]-1];
	
	T X1[] = {0.212340538, 0.59053313, 0.91141204, 0.212340538, 0.59053313, 0.91141204, 0.212340538, 0.59053313, 0.91141204}; 
        T Y1[] = {0.188409406, 0.523979067, 0.808694385, 0.106170269, 0.2952665677, 0.45570602024, 0.023931132, 0.0665540678, 0.1027176548}; //storing harcoded coordinates for gauss nodes on ref elements
	
	if(X == NULL) X = X1;
	if(Y == NULL) Y = Y1;
	for(int i=0; i < 9; i++)
	{
		x = X[i];
		y = Y[i];
		
		temp1 = phi1 + (phi2 - phi1)*x;
		temp2 = phi1 + (phi3 - phi1)*x;
		out[i] = temp1 + (temp2 - temp1)*y; 
	}
	
}


//---------------Function to calculate the Stokes double layer potential with density eta at target points trg_points-------------
//If self = true, trg_points are all vertices of triangulation i.e same as the source points
void DLP(int *tri, T *points, int ntri, int npoints, T *eta, bool self, T *trg_points, int size_trg, T *pot, MPI_Comm comm)  //pot is output, eta is denisty
{

	T trg_point[3];
	T K[9];
	T interpolated_eta[9]; // to store interpolated eta values at gauss nodes
        T X2[] = {0.023931132, 0.0665540678, 0.1027176548, 0.106170269, 0.2952665677, 0.45570602024, 0.188409406, 0.523979067, 0.808694385 }; //storing harcoded coordinates for gauss nodes on ref elements

	T Y2[] = {0.212340538, 0.59053313, 0.91141204, 0.212340538, 0.59053313, 0.91141204, 0.212340538, 0.59053313, 0.91141204}; 
       
	zero(pot, size_trg); //zero the array
	int rank, np;
	MPI_Comm_size(comm, &np);
	MPI_Comm_rank(comm, &rank);
	int start; // starting point of theglobal array that is local to this process
	if(rank < size_trg%np) start = (size_trg/np)*rank + rank;
	else start =  (size_trg/np)*rank + (size_trg%np);
	
	if(!self)
	{
		T src_tri[27]; // stores src coordinates obtained after mapping gaussian nodes on ref element to current element
		T src_refx[9]; //stores src x-coordinates on ref element obtained after inverse Duffy from duffy element
		T src_refy[9]; // same as above but y coordinates
		#pragma omp parallel for private(trg_point, K, interpolated_eta, src_tri, src_refx, src_refy)	
		for(int i=0; i < size_trg; i++)
		{
				
			trg_point[0] = trg_points[3*i];
			trg_point[1] = trg_points[3*i+1];
			trg_point[2] = trg_points[3*i+2];
			
			for(int j=0; j < ntri; j++)
			{
				
				//Integrate over j-th triangle
				if(inTriangle(tri, j, start+i+1) ) //If vertex i is in triangle j, start for offset to MPI proc, since size_trg and trg_points is local 
				{       //cout<<"Singular Integrate\n";
					//Singular Integrate
					singularTransform(src_refx, src_refy, NULL, NULL);
					//cout<<"L1\n";
					standardizeTransform(tri, j, points, src_tri, inTriangle(tri, j, start+i+1) ,src_refx, src_refy);
					//cout<<"L2\n";
					DLKernelSingular(trg_point, src_tri, 9, K, NULL); //includes singular jacobian 
				 	//cout<<"L3\n";
					interpolate(tri, j, eta, interpolated_eta, src_refx, src_refy);	
					//cout<<"L4\n";
				
					pot[i] += gaussInt(K, interpolated_eta, 3, NULL, NULL)*standardizeJac(tri, j, points);	
				
					//cout<<"L5\n";
					singularTransform(src_refx, src_refy, X2, Y2);
					//cout<<"L6\n";
					standardizeTransform(tri, j, points, src_tri, inTriangle(tri, j, start+i+1) ,src_refx, src_refy);
					//cout<<"L7\n";
					DLKernelSingular(trg_point, src_tri, 9, K, X2); //includes singular jacobian 
				 	//cout<<"L8\n";
					interpolate(tri, j, eta, interpolated_eta, src_refx, src_refy);	
					//cout<<"L9\n";
				
					pot[i] += gaussInt(K, interpolated_eta, 3, NULL, NULL)*standardizeJac(tri, j, points);	
					

				}
				else
				{
					//cout<<"Non singular Integrate\n";
					standardizeTransform(tri, j, points, src_tri, 0, NULL, NULL);
					DLKernel(trg_point, src_tri, 9, K);
				 	interpolate(tri, j, eta, interpolated_eta, NULL, NULL);	
				
					pot[i] += gaussInt(K, interpolated_eta, 3, NULL, NULL)*standardizeJac(tri, j, points);	
				
				}			
			}		
		}
	}
}

void print(T *arr, int size)
{
	for(int i=0; i < size; i++) cout<<" "<<arr[i];
}


void print(int *arr, int size)
{
	for(int i=0; i < size; i++) cout<<" "<<arr[i];
}

void computeError(T *pot, int size, MPI_Comm comm)
{
	int rank;
  	MPI_Comm_rank(comm, &rank);
	T maxerror_loc = 0;;
	T true_val = -0.5;
	for(int i = 0; i < size; i++)
	{
		if(abs(pot[i] - true_val) > maxerror_loc) maxerror_loc = abs(pot[i] - true_val);
	}
	T maxerror;
  	MPI_Reduce(&maxerror_loc, &maxerror, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	if (rank == 0)  std::cout << "The max error is " << maxerror << std::endl;	
}



//Petsc matvec
int mult(Mat M, Vec U, Vec Y)
{
	MPI_Comm comm;
	int np, myrank;
	PetscObjectGetComm((PetscObject) M , &comm);
	MPI_Comm_size(comm, &np);
	MPI_Comm_rank(comm, &myrank);
	void *dt = NULL;
	MatShellGetContext(M, &dt);
	Sargs *data = (Sargs *)dt;
	
	PetscErrorCode ierr;
	PetscInt U_size;
	const PetscScalar *U_ptr;
	ierr = VecGetLocalSize(U, &U_size);
	ierr = VecGetArrayRead(U, &U_ptr);

	PetscInt Y_size;
	PetscScalar *Y_ptr;
	ierr = VecGetLocalSize(Y, &Y_size);
	ierr = VecGetArrayRead(Y, &Y_ptr);

	
	for(int i=0; i < Y_size; i++)
	{
		Y_ptr[i] = U_ptr[i];
	}	

	ierr = VecRestoreArray(Y, &Y_ptr);
	return 0;
}

int main(int argc, char** argv)
{

	MPI_Init(NULL, NULL);
	//PetscErrorCode ierr;
	//PetscInitialize(&argc,&argv,0,help);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm comm = MPI_COMM_WORLD;			
	//Get the rank of process
	int rank, np;
	MPI_Comm_size(comm, &np);
	MPI_Comm_rank(comm, &rank);

	//------------------------------------------
	int ntri=395010;       //number of triangles
        int npoints = 197507;   //number of discretization points
	int size_trg = 197507;	
	//------------------------------------------

	int size_trg_loc = size_trg/np + (rank < size_trg%np ? 1 : 0); //number of trg points for each process
	

	int *tri; T *points, *eta, *pot_loc, *trg_points_loc; 
	tri = (int *) malloc(sizeof(int)*3*ntri);
	points = (T *)malloc(sizeof(T)*3*npoints);
	pot_loc = (T *)malloc(sizeof(T)* size_trg_loc);
	eta = (T *)malloc(sizeof(T)*npoints);
	trg_points_loc = (T *)malloc(sizeof(T)*3*size_trg_loc);
	for(int i=0; i < npoints; i++)
	{
		eta[i] = 1.0;
	}
	//--------------Read Triangulation---------------
	readTriangulation(tri, points, ntri, npoints);
	//------------------------------------------------
	
	//-------------Initializing static arguments-------------
	Sargs args;
	args.ntri = ntri;
	args.npoints = npoints;
	args.points = points;
	args.tri = tri;
	//------------------------------------------------------

	int start; // starting point of theglobal array that is local to this process
	if(rank < size_trg%np) start = (size_trg/np)*rank + rank;
	else start =  (size_trg/np)*rank + (size_trg%np);
	for(int i=0 ; i < size_trg_loc; i++)
	{
		trg_points_loc[3*i + 0] = points[start*3 + 3*i+0];
		trg_points_loc[3*i + 1] = points[start*3 + 3*i+1];
		trg_points_loc[3*i + 2] = points[start*3 + 3*i+2];
	}
	
	T t0, t1;
	MPI_Barrier(comm);
 	t0 = MPI_Wtime();				 
   	DLP(tri, points, ntri, npoints, eta, false, trg_points_loc, size_trg_loc, pot_loc, comm)	;
	MPI_Barrier(comm);
	t1 = MPI_Wtime();
	if(rank==0) print(pot_loc, size_trg_loc);
	computeError(pot_loc, size_trg_loc, comm);	
	if(!rank) cout<<"Time taken to apply matvec:"<<t1 - t0<<"\n";
	
		
	//------------PETSC SOLVE PART-------------------
	PetscInt m, n;
	PetscErrorCode ierr;
	m = size_trg_loc; n = size_trg;
  	Mat A; cout<<"\nHELLOOOOOO";
  	// Create Matrix. A
	MatCreateShell(comm,m,n,PETSC_DETERMINE,PETSC_DETERMINE,NULL,&A);
	MatShellSetOperation(A,MATOP_MULT,(void(*)(void))mult);
	MatShellSetContext(A, &args);

  	Vec f,x,b;
	// Create vectors
	VecCreateMPI(comm,n,PETSC_DETERMINE,&f);
	VecCreateMPI(comm,m,PETSC_DETERMINE,&b);
	VecCreateMPI(comm,n,PETSC_DETERMINE,&x); // Ax=b

	// Create Input Vector. f
	PetscInt f_size;
	ierr = VecGetLocalSize(f, &f_size);

  	PetscScalar *f_ptr;
	ierr = VecGetArray(f, &f_ptr);
	for(int i = 0 ; i < f_size; i++) f_ptr[i]=0.0;
	ierr = VecRestoreArray(f, &f_ptr);

  	// Create Input Vector.
	PetscInt b_size;
	ierr = VecGetLocalSize(b, &b_size);

  	PetscScalar *b_ptr;
   	PetscScalar *x_ptr;
	ierr = VecGetArray(b, &b_ptr);
	for(int i=0; i < b_size; i++) b_ptr[i] = 1.0;
	ierr = VecRestoreArray(b, &b_ptr);
	// Create solution vector
	ierr = VecDuplicate(f,&x); CHKERRQ(ierr);
	PetscInt x_size;
	ierr = VecGetLocalSize(x, &x_size);

	KSP ksp; ierr = KSPCreate(comm,&ksp); CHKERRQ(ierr);
	// Set operators. Here the matrix that defines the linear system
	// also serves as the preconditioning matrix.
	ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
	
	// Set runtime options
	KSPSetType(ksp, KSPGMRES);
	KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
	

	KSPSetTolerances(ksp, pow(10,-1*6), PETSC_DEFAULT,PETSC_DEFAULT, 500);
	KSPGMRESSetRestart(ksp, 500);
	MPI_Barrier(comm);
	t0 = MPI_Wtime();
	ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
	MPI_Barrier(comm);
	t1 = MPI_Wtime();

	if(!rank) cout<<"\nTime taken to solve system"<<t1 - t0<<"\n";
	
	PetscInt its;
	ierr = KSPGetIterationNumber(ksp,&its); CHKERRQ(ierr);
	ierr = PetscPrintf(comm,"Iterations %D\n",its); CHKERRQ(ierr);
	// Free work space.  All PETSc objects should be destroyed when they
	// are no longer needed.
	ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
	ierr = VecDestroy(&x);CHKERRQ(ierr);
	ierr = VecDestroy(&b);CHKERRQ(ierr);
	ierr = MatDestroy(&A);CHKERRQ(ierr);
	
	
	
	ierr = PetscFinalize();
        //-----------------------PETSC SOLVE ENDS-----------------------------
	MPI_Finalize();
	return 0;	

}

