// evolution for different kinds of conductance
void chain2(double *g,double *s3,double trise,double tdamp,double deltat,int nn){
/************************************************************
      subroutine chain2(ge,se1,se2,se3,trise,tdamp,deltat,nn)
!------------------------------------------------------------
!  Oct 2000: Updating difference of expon'tial w/o intracortical spikes
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      dimension ge(nn),se1(nn),se2(nn),se3(nn)
!------------------------------------------------------------*/
    double tr  = deltat/trise;
    double etr = exp(-tr);

    double td  = deltat/tdamp;
    double etd = exp(-td);

    double cst = trise/(tdamp - trise) * (etd - etr);
	  
	  for(int i = 0; i < nn; i++){
		  g[i]  =  g[i] * etd + cst * s3[i];
		  s3[i] =  s3[i] * etr;
	  }
    return ;
}

void chain(double *g,double *s3,double tau,double deltat,int nn){
/************************************************************
      subroutine chain2(ge,se1,se2,se3,trise,tdamp,deltat,nn)
!------------------------------------------------------------
!  Oct 2000: Updating difference of expon'tial w/o intracortical spikes
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      dimension ge(nn),se1(nn),se2(nn),se3(nn)
!------------------------------------------------------------*/
    double te  = deltat/tau;
    double ete = exp(-te);

    double cst = te * (ete);
	  
	  for(int i = 0; i < nn; i++){
		  g[i]  =  g[i] * ete + cst * s3[i];
		  s3[i] =  s3[i] * ete;
	  }
    return;
}

void conduc(double *g,double *s3,double tau,double deltat,int nn){
/************************************************************
      subroutine chain2(ge,se1,se2,se3,trise,tdamp,deltat,nn)
!------------------------------------------------------------
!  Oct 2000: Updating difference of expon'tial w/o intracortical spikes
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      dimension ge(nn),se1(nn),se2(nn),se3(nn)
!------------------------------------------------------------*/
    double te  = deltat/tau;
    double ete = exp(-te);

    double cst = te * (ete);
	  
	  for(int i = 0; i < nn; i++){
		  *(g+i)  =  *(g+i) * ete + cst * (*(s3+i));
		  (*(s3+i)) =  (*(s3+i)) * ete;
	  }
    return;
}