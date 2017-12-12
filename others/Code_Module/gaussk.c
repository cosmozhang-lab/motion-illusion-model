// function to calculate SR/LR recurrent connections
/*-------------------------------------------------------------
      subroutine gaussk(prefct,bb,aa,nx,ny,al2,dx2,dy2,fglobal)
!------------------------------------------------------------
!  Oct 2000: modified for chain (do NOT fourier transform)
!
!    bb : gaussian kernel on exit
!    aa : work array
!    bb = prefct/al2 * exp(-dd/al2), dd = (i-1)*(i-1)*dx2 + (j-1)*(j-1)*dy2
!         Gaussian centered at (1,1)
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      dimension aa(nx*ny),bb(nx,ny)
!------------------------------------------------------------*/
void gaussk(double prefct,double *bb,double *aa,int nx,int ny,double al2,double dx2,double dy2,double fglobal)
{
	int nhx = nx/2;
  int nhy = ny/2;
  double cst = sqrt(dx2*dy2)/nx/ny/4.0/atan(1.0);
  double dd;
  double **btemp = new double*[nx];
  for(int i = 0;i<nx;i++){
	  btemp[i]=new double[ny];
  }
  double sum = 0.0;
  for(int j =0;j<ny;j++){                    // fortran is column cluster
	for( int i = 0;i<nx;i++){
		dd = (i-nhx)*(i-nhx)*dx2 + (j-nhy)*(j-nhy)*dy2;
		//dd = (i-nhx-1)*(i-nhx-1)*dx2 + (j-nhy-1)*(j-nhy-1)*dy2;  fortran i, j from 1 to ... c from 0 nhx is the same
		btemp[i][j] = cst/al2*exp(-dd/al2);
	  }
  }

  for(int j = 0;j < ny;j++){
	  for(int i = 0;i < nx;i++){
		  //int jj     = (j-1)*nx + i;  jj e
		  int jj     = (j)*nx + i;
		  //int ii     = (nx*ny+jj-(nx*ny/2+nx/2+1))%(nx*ny) + 1;// aa[nx*ny] e also ,when j,i= ini = 1,jj =1 (ini) while in c, jj ini from0
		  // j=i=0 jj =0 fortran remain nxny,while c do the same(jj=0,1-0)
		  int ii     = (nx*ny+jj-(nx*ny/2+nx/2))%(nx*ny) ;      // ii start from 0;
		  aa[ii] = btemp[i][j];
		  sum    = sum + aa[ii];
	  }
  }
  //ccc = sum*nx*ny;
  for(int j = 0;j < ny;j++){
	  for( int i = 0;i < nx;i++){
		  //int jj       = (j-1)*nx + i;
		  int jj       = (j)*nx + i;
		  btemp[i][j] = prefct*aa[jj]/sum;
		  aa[jj]   = prefct/nx/ny;
		  bb[jj] = (1-fglobal)*btemp[i][j] + fglobal*aa[jj];
	  }
  }
}
