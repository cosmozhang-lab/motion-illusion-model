// function for updating different types conductance
void update(double t){
/*************************************************************
       subroutine update(conduc,nspike,ispike,tspike,dt,t,myid)
!------------------------------------------------------------
!  Update chain for excitatory & inhibitory after spikes
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
      parameter ( nmap = 32 )
      dimension ge(nmax),se1(nmax),se2(nmax),se3(nmax)
      dimension gf(nmax),sf1(nmax),sf2(nmax),sf3(nmax)
      dimension gi(nmax),si1(nmax),si2(nmax),si3(nmax)
      dimension gj(nmax),sj1(nmax),sj2(nmax),sj3(nmax)
      dimension gep(nmax),se1p(nmax),se2p(nmax),se3p(nmax)
      dimension gfp(nmax),sf1p(nmax),sf2p(nmax),sf3p(nmax)
      dimension gip(nmax),si1p(nmax),si2p(nmax),si3p(nmax)
      dimension gjp(nmax),sj1p(nmax),sj2p(nmax),sj3p(nmax)
      dimension a_ee(nmax),a_ei(nmax),a_ie(nmax),a_ii(nmax)
      dimension tspike(nmax),ispike(nmax),nlgni(nmax)
      dimension pspike(nmax)
      dimension icnntvy(nmax,nmap),indmap(nmax)
      logical excite(nmax)
      character*80 fsps
      common / chaine / ge,se1,se2,se3,gep,se1p,se2p,se3p
      common / chaine2/ gf,sf1,sf2,sf3,gfp,sf1p,sf2p,sf3p
      common / chaini / gi,si1,si2,si3,gip,si1p,si2p,si3p
      common / chainj / gj,sj1,sj2,sj3,gjp,sj1p,sj2p,sj3p
      common / tconst / tau_e,tau_i,tau0,tau1,tau2,tnrise,tndamp
      common /  NMDA  / fnmdat,fnmdac,fgaba
      common / kernel / a_ee,a_ei,a_ie,a_ii
      common / neuron / excite,nlgni
      common / ttotal / tfinal,twindow
      common / synapt / pconn
      common /   isi  / pspike,nsptot,isptot,nessp,necsp,nissp,nicsp
      common / sparse / icnntvy,indmap
      common / filenm / fsps
      external conduc
      data iword  / 8 /
      real*4 fnsp
!------------------------------------------------------------*/
	if (nspike>0){
		disort();		  
	for(int i = 0; i < nmax; i++){
		ge[i]  = gep[i];
		se3[i] = se3p[i];
		gf[i]  = gfp[i];
		sf3[i] = sf3p[i];
		gi[i]  = gip[i];
		si3[i] = si3p[i];
		gj[i]  = gjp[i];
		sj3[i] = sj3p[i];
		
		gel[i]  = gelp[i];
		sel3[i] = sel3p[i];
		gfl[i]  = gflp[i];
		sfl3[i] = sfl3p[i];
	}
	double taueratio = tau_e/tnrise;
	double tauiratio = tau_i/tau2;
	double tt;
	double cnntvy;


	tt = tspike[0];
	for( int j = 0; j < nspike; j++){
		conduc(ge,se3,tau_e,tt,nmax);
		chain2(gf,sf3,tnrise,tndamp,tt,nmax);
		conduc(gi,si3,tau_i,tt,nmax);
		conduc(gj,sj3,tau2,tt,nmax);
		/* update all Depression factors 
		Depression(dlo,tau_d,tt,nmax);
	  Depression(slo,tau_s,tt,nmax);
	  */

		int ij = ispike[j]; 
		/* ij is pre-synaptic spike, induce short term synaptic depression;
		 * if excite[ij] == true, then will lead to change in gexc,excitatory 
		 * conductance will be influenced by short term depression(inhibitory won't)*/

//  calculate delta-function amplitudes given ispike[j]
		if (excite[ij]){
			/* if pre-synape induce change in excitatory conductance
			 * then update correspond Depression factor 
			slo[ij]   = slo[ij] * drs;
			dlo[ij]   = dlo[ij] * drd;
			*/
			for(int i = 0; i < nmax; i++){
				int ii = (nmax+i-ij)%nmax ;								//fortran use 1 as start, c use 0 as start
				cnntvy = icnntvy[ii][indmap[i]]*1.0;
				if(clusthyp[ij] == clusthyp[i]{
  				if (cnntvy >0.5){					
  					if (excite[i]){
  						se3[i] = se3[i] + a_ee[ii] * cnntvy;
  						sf3[i] = sf3[i] + a_ee[ii] * taueratio * cnntvy;
  					}else{
  						se3[i] = se3[i] + a_ie[ii] * cnntvy;
  						sf3[i] = sf3[i] + a_ie[ii] * taueratio * cnntvy;
  					}
  				}
  			}else{
  				if(clustorien[ij] == clustorien[i]){
  					if (cnntvy >0.5){					
    					if (excite[i]){
    						sel3[i] = sel3[i] + lr_ee[ii] * cnntvy;
    						sfl3[i] = sfl3[i] + lr_ee[ii] * taueratio * cnntvy;
    					}else{
    						sel3[i] = sel3[i] + lr_ie[ii] * cnntvy;
    						sfl3[i] = sfl3[i] + lr_ie[ii] * taueratio * cnntvy;
    					}
    				}
    			}
    		}
    	}
		}else{
			for(int i = 0; i < nmax; i++){
				//int ii = (nmax+i-ij)%nmax + 1;
				int ii = (nmax+i-ij)%nmax ;								//fortran use 1 as start, c use 0 as start
				cnntvy = icnntvy[ii][indmap[i]]*1.0;
				if (cnntvy >0.5){
					if (excite[i]){
						si3[i] = si3[i] + a_ei[ii] * cnntvy;
						sj3[i] = sj3[i] + a_ei[ii] * tauiratio * cnntvy;
					}else{
						si3[i] = si3[i] + a_ii[ii] * cnntvy;
						sj3[i] = sj3[i] + a_ii[ii] * tauiratio * cnntvy;
					}
				}
			}
		}
//------------------------------------Onto next subinterval!
	  tt = tspike[j+1] - tspike[j];
//----------------------------------------------------------
	}
//-----------------Update chain between last spike & next time
	conduc(ge,se3,tau_e,dt-tspike[nspike],nmax);
	chain2(gf,sf3,tnrise,tndamp,dt-tspike[nspike],nmax);
	
	conduc(gel,sel3,tau_e,dt-tspike[nspike],nmax);
	chain2(gfl,sfl3,tnrise,tndamp,dt-tspike[nspike],nmax);
	
	conduc(gi,si3,tau_i,dt-tspike[nspike],nmax);
	conduc(gj,sj3,tau2,dt-tspike[nspike],nmax);
//	Depression(dlo,tau_d,dt-tspike[nspike],nmax);
//	Depression(slo,tau_s,dt-tspike[nspike],nmax);
	
  }
      for(int i = 0; i < nmax; i++){
		  gep[i]  =  ge[i];
		  se3p[i] = se3[i];
		  gfp[i]  =  gf[i];
		  sf3p[i] = sf3[i];
		  
		  gelp[i]  =  gel[i];
		  sel3p[i] = sel3[i];
		  gflp[i]  =  gfl[i];
		  sfl3p[i] = sfl3[i];
		  
		  gip[i]  =  gi[i];
		  si3p[i] = si3[i];
		  gjp[i]  =  gj[i];
		  sj3p[i] = sj3[i];
//		  dlop[i] = dlo[i];
//		  slop[i] = slo[i];
	  }
    return;
}