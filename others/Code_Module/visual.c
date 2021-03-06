// function for LGN inputs complex version
void visual(double frtlgn0,double frtinhE, double ciE0,double frtinhI,double ciI0,double t,unsigned iseed){
/************************************************************
      subroutine visual(frtlgn0,cond0,frtinhE,ciE0,frtinhI,ciI0,t,dt,iseed)
!------------------------------------------------------------
!     Also generate noise (1-1) given firing rates
!
!     Generates LGN spike times using Poisson process 
!	with time-dependent firing rate a function of
!	visual stimulus
!------------------------------------------------------------*/
/*     IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj , nspmax = 50 )
      parameter ( nlx = 48 , nly = 64 , nlgn = nlx*nly , maxlgn = 60 )
      dimension glo(nlgn), slo1(nlgn), slo2(nlgn), slo3(nlgn)
      dimension glon(nlgn),slon1(nlgn),slon2(nlgn),slon3(nlgn)
      dimension glop(nlgn),slo1p(nlgn),slo2p(nlgn),slo3p(nlgn)
      dimension glonp(nlgn),slon1p(nlgn),slon2p(nlgn),slon3p(nlgn)
      dimension glf(nlgn), slf1(nlgn), slf2(nlgn), slf3(nlgn)
      dimension glfn(nlgn),slfn1(nlgn),slfn2(nlgn),slfn3(nlgn)
      dimension glfp(nlgn),slf1p(nlgn),slf2p(nlgn),slf3p(nlgn)
      dimension glfnp(nlgn),slfn1p(nlgn),slfn2p(nlgn),slfn3p(nlgn)
      dimension gx(nmax), sx1(nmax), sx2(nmax), sx3(nmax)
      dimension gy(nmax), sy1(nmax), sy2(nmax), sy3(nmax)
      dimension gxp(nmax),sx1p(nmax),sx2p(nmax),sx3p(nmax)
      dimension gyp(nmax),sy1p(nmax),sy2p(nmax),sy3p(nmax)
      dimension gn(nmax), sn1(nmax), sn2(nmax), sn3(nmax)
      dimension gnp(nmax),sn1p(nmax),sn2p(nmax),sn3p(nmax)
      dimension gm(nmax), sm1(nmax), sm2(nmax), sm3(nmax)
      dimension gmp(nmax),sm1p(nmax),sm2p(nmax),sm3p(nmax)
      dimension ampcon(nlx*nly),ampson(nlx*nly)
      dimension ampcof(nlx*nly),ampsof(nlx*nly)
      dimension pspon(nlx*nly),pspoff(nlx*nly)
      dimension pspexc(nmax),pspinh(nmax)
      dimension nlgni(nmax)
      dimension contr(200)
      logical excite(nmax)
      common / neuron / excite,nlgni
      common / lgnsps / pspon,pspoff
      common /  psps  / pspexc,pspinh
      common / chaino / glo,slo1,slo2,slo3,glop,slo1p,slo2p,slo3p
      common / chaino2/ glon,slon1,slon2,slon3, &
		glonp,slon1p,slon2p,slon3p
      common / chainf / glf,slf1,slf2,slf3,glfp,slf1p,slf2p,slf3p
      common / chainf2/ glfn,slfn1,slfn2,slfn3, &
		glfnp,slfn1p,slfn2p,slfn3p
      common / chainx / gx,sx1,sx2,sx3,gxp,sx1p,sx2p,sx3p
      common / chainy / gy,sy1,sy2,sy3,gyp,sy1p,sy2p,sy3p
      common / chainn / gn,sn1,sn2,sn3,gnp,sn1p,sn2p,sn3p
      common / chainm / gm,sm1,sm2,sm3,gmp,sm1p,sm2p,sm3p
      common / tconst / tau_e,tau_i,tau0,tau1,tau2,tnrise,tndamp
      common /  NMDA  / fnmdat,fnmdac,fgaba
      common / lgnrfs / ampson,ampsof,ampcon,ampcof
      common / conrev / omega,gkx,gky,gphi,tstart,contrast
      common / consat / contr*/
	  //srand(iseed);
    double omegat = omega*(t-tstart);
	  double frate,rannum,dtt,tlocal,te,ete,tr,etr,td,etd,ti,eti,tj,etj,tnext,ci0,cst;
	  double dte,ste;
	  // case 1
    if (t < tstart) frate = frtlgn0;
    // case 2
	  if (frtlgn0 > 0.0){
		  for(int i = 0;i < nlgn; i++){
			  if (t>pspon[i]+0.0015){
/*-------------Find firing rate as function of visual stimulus
! For CONTRAST REVERSAL, need only amplitude of sinusoid
!------------------------------------------------------------*/
				  if ((t>0.0)&&(t<tstart)){
					  frate = frtlgn0 + (t/tstart) * 
						  (ampson[i]*sin(omegat) + ampcon[i]*cos(omegat));
				  }
				  if (t > tstart){
					  frate = frtlgn0 + ampson[i]*sin(omegat) + ampcon[i]*cos(omegat);
				  }
//------------------------Compute spikes only if non-zero rate
				  if (frate > 0 ){
/*------------------------------------------------------------
! Contrast Saturation
!------------------------------------------------------------
!	  ifrate = min(int(frate)*10.d0/frtlgn0,199)
!	  r2  = contr(ifrate+1)
!	  r1  = contr(ifrate)
!	  satur  = r1 + (r2-r1) * (frate*10.d0/frtlgn0-ifrate)
! 	  if (i==1) print *,sngl(frate),sngl(satur),ifrate
!	  frate  = satur*frtlgn0/10.d0
!------------------------------------------------------------*/
					  //srand(iseed);
				   	rannum = rand()/(RAND_MAX+1.0);
					  if (rannum < dt*frate){
						  /* my calculation method
						  spiketmp = t + rannum/frate;
						  tint     = spiketmp-pspon[i];

						  glo[i]  = glop[i] ;
						  slo3[i] = slo3p[i];
						  glon[i]  = glonp[i] ;
						  slon3[i] = slon3p[i];
						  dtt = rannum/frate;
						  te  = tint/tau_e;
						  ete = exp(-te);
						  glo[i]  = (glo[i]  + slo3[i]) * ete;
						  slo3[i] = (slo3[i]          ) * ete + cond0;
						  
						  tr  = tint/tnrise;
						  etr = exp(-tr);
						  td  = dtt/tndamp;
						  etd = exp(-td);
						  cst = tnrise/(tndamp - tnrise) * (etd - etr);
						  glon[i]  =  glon[i] * etd + cst * slon3[i];
						  slon3[i] = slon3[i] * etr + cond0*tau_e/tnrise;
						  // method above should be terminate at spiketmp, because updated pspon[i]=t+rannum/frate,
						  // if t+dt-spiketmp is processed, then at next spike time, the time interval which was started at pspon[i] have some
						  // problem. */

						  /* original calculation method */
						  pspon[i] = t + rannum/frate;
						  
						  glo[i]  = glop[i] ;
						  slo3[i] = slo3p[i];
						  glon[i]  = glonp[i] ;
						  slon3[i] = slon3p[i];						  

						  dtt = rannum/frate;
						  te  = dtt/tau_e;
						  ete = exp(-te);
						  glo[i]  = (glo[i]  + slo3[i]) * ete;
						  slo3[i] = (slo3[i]          ) * ete + cond0;
							  
						  tr  = dtt/tnrise;
						  etr = exp(-tr);
						  td  = dtt/tndamp;
						  etd = exp(-td);
						  cst = tnrise/(tndamp - tnrise) * (etd - etr);
						  glon[i]  =  glon[i] * etd + cst * slon3[i];
						  slon3[i] = slon3[i] * etr + cond0*tau_e/tnrise;
						  
						  te  = (dt-dtt)/tau_e;
						  ete = exp(-te);
						  glo[i]  = (glo[i]  + slo3[i]*te)*ete;
						  slo3[i] = (slo3[i]             )*ete;
						  
						  tr  = (dt-dtt)/tnrise;
						  etr = exp(-tr);
						  td  = (dt-dtt)/tndamp;
						  etd = exp(-td);
						  cst = tnrise/(tndamp - tnrise) * (etd - etr);
						  glon[i]  =  glon[i] * etd + cst * slon3[i];
						  slon3[i] = slon3[i] * etr;					  

					  }else{												// !!!!! This is the modified version of function visual,since if do not have 
						  glo[i]  = glop[i] ;								// LGN spike, conductance of LGN decays as well
						  slo3[i] = slo3p[i];
						  glon[i]  = glonp[i] ;
						  slon3[i] = slon3p[i];				  

						  dtt = dt;
						  te  = dtt/tau_e;
						  ete = exp(-te);
						  glo[i]  = (glo[i]  + slo3[i]) * ete;
						  slo3[i] = (slo3[i]          ) * ete;	  
						  
						  tr  = dtt/tnrise;
						  etr = exp(-tr);
						  td  = dtt/tndamp;
						  etd = exp(-td);
						  cst = tnrise/(tndamp - tnrise) * (etd - etr);
						  glon[i]  =  glon[i] * etd + cst * slon3[i];
						  slon3[i] = slon3[i] * etr;
					  }

				  }
			  } 
			  glop[i]  = glo[i];
			  slo3p[i] = slo3[i];
			  glonp[i] = glon[i];
			  slon3p[i] = slon3[i];  
		  } 
//---------------------------------Now off-centered LGN cells
		  //srand(iseed);
		  for(int i = 0;i < nlgn; i++){
			  if (t > pspof[i]+0.0015){
				  if ((t > 0.0)&&(t < tstart)){
					  frate = frtlgn0 + (t/tstart) * 
						  (ampsof[i]*sin(omegat) + ampcof[i]*cos(omegat));
				  }
				  if (t > tstart) {
					  frate = frtlgn0 + ampsof[i]*sin(omegat) + ampcof[i]*cos(omegat);
				  }
//------------------------------------------------------------
				  if (frate > 0.0){
/*------------------------------------------------------------
! Contrast Saturation
!------------------------------------------------------------
!	  ifrate = min(int(frate)*10.d0/frtlgn0,199)
!	  r2  = contr(ifrate+1)
!	  r1  = contr(ifrate)
!	  satur  = r1 + (r2-r1) * (frate*10.d0/frtlgn0-ifrate)
!c	  if (i==1) print *,sngl(frate),sngl(satur),ifrate
!	  frate  = satur*frtlgn0/10.d0
!------------------------------------------------------------*/
					  //srand(iseed);
				   	rannum = rand()/(RAND_MAX+1.0);
					  if (rannum < dt*frate){
						  pspof[i] = t + rannum/frate;
						  glf[i]   = glfp[i] ;
						  slf3[i]  = slf3p[i];
						  glfn[i]  = glfnp[i];
						  slfn3[i] = slfn3p[i];			  
						  
						  dtt      = rannum/frate;
						  te       = dtt/tau_e;
						  ete      = exp(-te);
						  glf[i]   = (glf[i]  + slf3[i]*te) * ete;
						  slf3[i]  = (slf3[i]             ) * ete + cond0;

						  tr  = dtt/tnrise;
						  etr = exp(-tr);
						  td  = dtt/tndamp;
						  etd = exp(-td);
						  cst = tnrise/(tndamp - tnrise) * (etd - etr);
						  glfn[i]  = glfn[i]  * etd + cst * slfn3[i];
						  slfn3[i] = slfn3[i] * etr + cond0*tau_e/tnrise;
						  
						  te  = (dt-dtt)/tau_e;
						  ete = exp(-te);
						  glf[i]  = (glf[i]  + slf3[i]*te)*ete;
						  slf3[i] = (slf3[i]             )*ete ;
						  
						  tr  = (dt-dtt)/tnrise;
						  etr = exp(-tr);
						  td  = (dt-dtt)/tndamp;
						  etd = exp(-td);
						  cst = tnrise/(tndamp - tnrise) * (etd - etr);
						  glfn[i]  = glfn[i] * etd + cst * slfn3[i];
						  slfn3[i] = slfn3[i] * etr;

					  }else{
						  glf[i]   = glfp[i] ;
						  slf3[i]  = slf3p[i];
						  glfn[i]  = glfnp[i];
						  slfn3[i] = slfn3p[i];

						  dtt      = dt;
						  te       = dtt/tau_e;
						  ete      = exp(-te);
						  glf[i]   = (glf[i]  + slf3[i]*te) * ete;
						  slf3[i]  = (slf3[i]             ) * ete;

						  tr  = dtt/tnrise;
						  etr = exp(-tr);
						  td  = dtt/tndamp;
						  etd = exp(-td);
						  cst = tnrise/(tndamp - tnrise) * (etd - etr);
						  glfn[i]  =  glfn[i] * etd + cst * slfn3[i];
						  slfn3[i] = slfn3[i] * etr ;
					  }
					  // this 'else' is post-added,if we don't have this 'else',then next 'conduc''chain2'...can be used to calculate the 
					  // appropriate ideal gl in advance.
					  // Eg. spike between t~t+dt and then record g(t+dt) in p,use chain2 to calculate ideal decay in t+dt~t+2dt,now g is refreshed to be
					  // g(t+dt)*exp--> g(t+2dt)(while p is still g(t+dt)), if do not fire at t+dt~t+2dt,then g-->pre(refresh)==g(t+2dt), the exact value!
				  }
			  }
			  glfp[i]  = glf[i] ;
			  slf3p[i] = slf3[i];
			  glfnp[i]  = glfn[i] ;
			  slfn3p[i] = slfn3[i];
		  }
		}
//----------------------------------Now inhibitory noise units
		//frate = frtinhE;
		if (frate > 0.0){
			for(int i = 0 ; i < nmax; i++){
				if ( excite[i] ) {
					frate = frtinhE;
					ci0 = ciE0;
				}else{
					frate = frtinhI;
					ci0 = ciI0;
				}
				
				tlocal  = t;
				gn[i]   = gnp[i];
				sn3[i]  = sn3p[i];
				gm[i]   = gmp[i];
				sm3[i]  = sm3p[i];
				//102    continue;
/* pay attention to this "while", do while and while are processed in different order,and may obtain different result!*/
				while(t+dt > pspinh[i]){
					dtt = pspinh[i] - tlocal;
					
					ti  = dtt/tau_i;
					eti = exp(-ti);
					tj  = dtt/tau2;
					etj = exp(-tj);

					gn[i]  = (gn[i]  + sn3[i]*ti)*eti;
					sn3[i] = (sn3[i]            )*eti + ci0/tau_i;
					
					gm[i]  = (gm[i]  + sm3[i]*tj)*etj;
					sm3[i] = (sm3[i]            )*etj + ci0/tau2;
					
					tlocal = pspinh[i];
					//srand(iseed);
				  rannum = rand()/(RAND_MAX+1.0);
					tnext  = -log(rannum)/frate;
					pspinh[i] = pspinh[i] + tnext;
				}
				
				dtt = t + dt - tlocal;
				
				ti  = dtt/tau_i;
				eti = exp(-ti);
				tj  = dtt/tau2;
				etj = exp(-tj);
				
				gn[i]  = (gn[i]  + sn3[i]*ti)*eti;
				sn3[i] = (sn3[i]            )*eti;
				gm[i]  = (gm[i]  + sm3[i]*tj)*etj;
				sm3[i] = (sm3[i]            )*etj;
				gnp[i]  = gn[i];
				sn3p[i] = sn3[i];
				gmp[i]  = gm[i];
				sm3p[i] = sm3[i];
			}
		}
      return;
}
