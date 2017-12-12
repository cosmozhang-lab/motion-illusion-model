// generate mapping 
void genmap(double pconnect,unsigned iseed){
/************************************************************
      subroutine genmap(icnntvy,indmap,pconnect,iseed,myid)
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
      parameter ( nlx = 48 , nly = 64 , nlgn = nlx*nly)
      parameter ( nn = nmax , nmap = 32 )
!------------------------------------------------------------
!  Generate Postsynaptic Target Map
!------------------------------------------------------------
      dimension icnntvy(nmax,nmap),ind(nmax),indmap(nmax)
      data iword / 8 /
!------------------------------------------------------------*/
	double nx1 = ni/2.25 + 1;
	double ny1 = nj/2.25 + 1;
	double nx2 = ni - ni/2.25 - 1;
	double ny2 = nj - nj/2.25 - 1;
	int   *ind = new int[nmax];
//      nx1 = ni/4 + 1
//      ny1 = nj/4 + 1
//      nx2 = ni - ni/4 - 1
//      ny2 = nj - nj/4 - 1
	
	for(int i = 0;i<nmap;i++){
		for(int j = 0; j < nmax; j++){
			icnntvy[j][i] = 0;
		}
		
		int nconn = (int)(pconnect*nmax);
		for(int j =0;j < nconn ;j++){
			f1:	
			//ix = ran2(iseed) * ni + 1;
			//iy = ran2(iseed) * nj + 1;
			//int ix = (rand()/(RAND_MAX+1.0))*ni+1;
			//int iy = (rand()/(RAND_MAX+1.0))*nj+1;
			int ix = (rand()/(RAND_MAX+1.0))*ni;   // ix, iy start from 0
			int iy = (rand()/(RAND_MAX+1.0))*nj;
			if ((ix>nx1)&&(ix<nx2)) goto f1;
			if ((iy>ny1)&&(iy<ny2)) goto f1;
			ind[j] = (iy)*ni + ix;                // ix, iy start from 0, do not use iy-1
			for(int jj = 0;jj < j; jj++){         // fortran 1,j-1 all index less than j
				if (ind[j]==ind[jj]) goto f1;
			}
		}
		for(int j = 0;j < nconn; j++){
			icnntvy[ind[j]][i] = 1;
		}
	}
	for(int j = 0; j < nmax; j++){
		ind[j]    = (rand()/(RAND_MAX+1.0))*nmax;
		indmap[j] = ind[j]%nmap;			// the same as mentioned above, if we have ind[j] = nmap-1, then may obtain nmap,icnntvy[][nmap] exceed the boundary
	}
/*!------------------------------------------------------------
      if (myid .eq. 0) then
      open(20,file='map.dat',status='new',form='unformatted', &
		access='direct',recl=nmax*iword/2)
!      print *,'Writing Map Index'
!      write(20,rec=1) indmap
!      print *,'Writing Connectivity Maps'
      do i=1,nmap
	write(20,rec=i+1) (icnntvy(j,i),j=1,nmax)
      enddo
      close(20)
      endif
!------------------------------------------------------------*/
      return;
}