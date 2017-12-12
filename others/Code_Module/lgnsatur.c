// function for satuation not use
void lgnsatur(){
/************************************************************
      subroutine lgnsatur
!------------------------------------------------------------
! Melinda's LGN contrast saturation function
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      dimension contr(200)
!------------------------------------------------------------
      common / consat / contr
!------------------------------------------------------------*/
	double c2 = 0.09827;
	double c3 = -0.001589;
	double pi = twopi/2.0;
	
	double satmax = 60;
	double join=30;
	double d1 = 40;
	double d4 = satmax-d1*pi/2;
	double zz = tan((join - d4)/d1) + pi;
	double d2 = (zz*zz+1)/d1;
	double d3 = zz - join * d2;

	for( int i = 0; i < 40; i++){
		double rr = i;
		contr[i] = c3*(pow(rr,3)) + c2*(pow(rr,2));
	}

    for( int i = 40; i < 200; i++){  
		double rr = i;
		contr[i] = d1 * atan(d2*rr + d3) + d4;
	}
      return;
}