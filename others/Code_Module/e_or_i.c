// function to tell E/I population
void e_or_i(unsigned iseed){
/************************************************************
      subroutine e_or_i(excite,exc,iseed)
!------------------------------------------------------------
!  Set up excitatory/inhibitory tag
!  One Quarter of population is inhibitory
!    regular (or random) lattice
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
      logical excite(ni,nj)
      dimension exc(ni,nj)
!------------------------------------------------------------*/
	for( int j = 0 ; j < nj; j++){
		for( int i= 0; i < ni; i++){
			int inde     = j*ni+i;
			excite[inde] = true;
			exc[inde] = 1;
			if ((j%2==0)&&(i%2==1)){   //	if ((mod(j,2).eq.1).and.(mod(i,2).eq.0)) excite(i,j) = .false.
/*
Uncomment line below AND comment line above to make inhibitory
locations random
if (ran2(iseed).lt.0.25D0) then
*/			excite[inde] = false;
				exc[inde] = 0;
			}
		}
	}
	return;
}