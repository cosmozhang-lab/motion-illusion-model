// RK2 MOST IMPORTANT
void rk2_lif(double t,double dt){
/* core codes for modified Runge- Kutta*/
/*------------------------------------------------------------
!  First update total conductance & total current
! { keep i.c.'s of chain, to re-use if any neuron spikes
!------------------------------------------------------------*/
    double dt2 = dt/2.0;double tt = 0;double dtsp = 0;double omegat = 0;
    nspike = 0;

	  for(int i = 0;i < nmax; i++){
	  	// reset LGN conductance to zero
		  gl[i]	  = 0.0; 
		  // calculate LGN conductance (ON/OFF seperately)
		  for( int j =0; j <nonlgn[i];j++){
			  gl[i] = gl[i] + (1-fnmdat)*glo[ionlgn[i][j]]
			  + fnmdat*glon[ionlgn[i][j]];
		  }
		  for( int j =0; j <noflgn[i];j++){
			  gl[i] = gl[i] + (1-fnmdat)*glf[ioflgn[i][j]]
			  + fnmdat*glfn[ioflgn[i][j]];
		  }
		  // effective LGN conductance
		  gl[i]    = gl[i]*nlgneff[i]/20.0;
		  if ( excite[i] ){
			  ginhib  = ((1-fgaba)*gi[i] + fgaba*gj[i])*sei[i];
			  gexcit  = ((1-fnmdac)*ge[i] + fnmdac*gf[i])*see[i] +((1-fnmdalc)*gel[i] + fnmdalc*gfl[i])*lee;
		  }else{
			  ginhib  = ((1-fgaba)*gi[i] + fgaba*gj[i])*sii[i];
			  gexcit  = ((1-fnmdac)*ge[i] + fnmdac*gf[i])*sie[i] +((1-fnmdalc)*gel[i] + fnmdalc*gfl[i])*lie;
		  }
		  ginoise   = (1-fgaba)*gn[i] + fgaba*gm[i];
		  genoise   = (1-fnmdat)*gx[i] + fnmdat*gy[i];
//		  if(SCind[i]>0.4){
//			  alpha0[i] = -gleak - gl[i] - gexcit - ginhib - genoise - ginoise;
//			  beta0[i]  = (gl[i] + gexcit + genoise)*vexcit
//				  + (ginoise + ginhib)*vinhib;
//		  }else{
//			  alpha0[i] = -gleak - gexcit - ginhib - genoise - ginoise;
//			  beta0[i]  = (+ gexcit + genoise)*vexcit
//				  + (ginoise + ginhib)*vinhib;
//		  }
			alpha0[i] = -gleak - gl[i] - gexcit - ginhib - genoise - ginoise;
			beta0[i]  = (gl[i] + gexcit + genoise)*vexcit
				  + (ginoise + ginhib)*vinhib;
	  }

      conduc(glo,slo3,tau_e,dt,nlgn);
      chain2(glon,slon3,tnrise,tndamp,dt,nlgn);
      conduc(glf,slf3,tau_e,dt,nlgn);
      chain2(glfn,slfn3,tnrise,tndamp,dt,nlgn);
      conduc(ge,se3,tau_e,dt,nmax);
      chain2(gf,sf3,tnrise,tndamp,dt,nmax);
      conduc(gel,sel3,tau_e,dt,nmax);
      chain2(gfl,sfl3,tnrise,tndamp,dt,nmax);
      conduc(gi,si3,tau_i,dt,nmax);
      conduc(gj,sj3,tau2,dt,nmax);
      conduc(gx,sx3,tau_e,dt,nmax);
      chain2(gy,sy3,tnrise,tndamp,dt,nmax);
      conduc(gn,sn3,tau_i,dt,nmax);
      conduc(gm,sm3,tau2,dt,nmax);
      double suma  = 0.0;
      double sumb  = 0.0;
      double suml  = 0.0;
      double sume  = 0.0;
      double sumi  = 0.0;
      double sumen = 0.0;
      double sumin = 0.0;
	  double a0,b0,a1,b1;
	  for(int i = 0;i < nmax; i++){
		  gl[i]	  = 0.0; 
		  for( int j =0; j <nonlgn[i];j++){
			  gl[i] = gl[i] + (1-fnmdat)*glo[ionlgn[i][j]]
			  + fnmdat*glon[ionlgn[i][j]];
		  }
		  for( int j =0; j <noflgn[i];j++){
			  gl[i] = gl[i] + (1-fnmdat)*glf[ioflgn[i][j]]
			  + fnmdat*glfn[ioflgn[i][j]];
		  }
		  gl[i]    = gl[i]*nlgneff[i]/20.0;
		  
		  if ( excite[i] ){
			  ginhib  = ((1-fgaba)*gi[i] + fgaba*gj[i])*sei[i];
			  gexcit  = ((1-fnmdac)*ge[i] + fnmdac*gf[i])*see[i] +((1-fnmdalc)*gel[i] + fnmdalc*gfl[i])*lee;
		  }else{
			  ginhib  = ((1-fgaba)*gi[i] + fgaba*gj[i])*sii[i];
			  gexcit  = ((1-fnmdac)*ge[i] + fnmdac*gf[i])*sie[i] +((1-fnmdalc)*gel[i] + fnmdalc*gfl[i])*lie;
		  }
		  ginoise   = (1-fgaba)*gn[i] + fgaba*gm[i];
		  genoise   = (1-fnmdat)*gx[i] + fnmdat*gy[i];
		  /* Simple cell and Complex cell*/
//		  if(SCind[i]>0.4){
//			  alpha1[i] = -gleak - gl[i] - gexcit - ginhib - genoise - ginoise;
//			  beta1[i]  = (gl[i] + gexcit + genoise)*vexcit 
//				  + (ginoise + ginhib)*vinhib;
//		  }else{
//			  alpha1[i] = -gleak - gexcit - ginhib - genoise - ginoise;
//			  beta1[i]  = (+ gexcit + genoise)*vexcit 
//				  + (ginoise + ginhib)*vinhib;
//		  }

		  suma  = suma - alpha1[i];
		  sumb  = sumb + beta1[i];
		  suml  = suml + gl[i];
		  sume  = sume + gexcit;
		  sumi  = sumi + ginhib;
		  sumen = sumen + genoise;
		  sumin = sumin + ginoise;
	  }
     for(int i = 0; i < nmax; i++){
     	if ( excite[i] ) {
			  tref = 0.003;
		  }else{
			  tref = 0.001;
		  }
		  a0   =  alpha0[i];
		  b0   =  beta0[i];
		  a1   =  alpha1[i];
		  b1   =  beta1[i];
		  
		  double fk1 = a0*v[i] + b0;
		  double fk2 = a1*(v[i]+dt*fk1) + b1;
		  vnew[i] =  v[i] + dt2*(fk1+fk2);
/*--------------------Evolve blocked potential
!        bk1 = a0*vb[i] + b0
!        bk2 = a1*(vb[i]+dt*bk1) + b1
!        vbnew[i] = vb[i] + dt2*(bk1+bk2)
!---------------------if neuron fires*/
		  if ((vnew[i]>vthres) &&((t+dt-pspike[i])>tref)){
/*------------------------------------------------------------
!    1. estimate spike time by cubic interpolation
!    2. calculate new init cond for the reset (extrapolate)
!    3. calculate vnew after spike (retake rk4 step)
!------------------------------------------------------------*/
			  dtsp = dt*(vthres-v[i])/(vnew[i]-v[i]);
			  vnew[i] = vreset;
			  //nspike = nspike + 1;  the nspike(th) spike means ()[spike-1] 
			  tspike[nspike] = dtsp;
			  ispike[nspike] = i;
			  nspike = nspike + 1;
			  nsptot = nsptot + 1;
			  if (!excite[i] ){
				  isptot = isptot + 1;
			  }
			  pspike[i] = t + dtsp;
/*------------------------------------------------------------
!  Construct histogram for spike rate
!------------------------------------------------------------*/
			  if (t > 2.0-dt) {
				  tirate = mod(t+dtsp,period);      // % can not be used on double float
				  nirate = int(tirate*100*0.25/period) + 1;
				  irate[i][nirate] = irate[i][nirate] + 1;
			  }/**/
//--------------------------endif neuron fires
		  }
/*------------------------------------------------------------
!  Refractory period
��so a neuron doesn't spike more than
��physically possible
!------------------------------------------------------------*/
		  if (pspike[i]>-1.0){
			  if ( excite[i] ){
				  tref = 0.003;
			  }else{
				  tref = 0.001;
			  }
			  if ((t+dt-pspike[i])<tref) {
				  vnew[i] = vreset;
			  }else if ((t+dt - pspike[i])<(tref+dt)){
				  tt = t+dt-pspike[i]-tref;
				  double vn   =  (vreset-tt*(b0+b1+b0*a1*dt)/2.0)/
					  (1.0+dtsp*(a0+a1+a0*a1*dt)/2.0);
				  fk1  =  a0*vn + b0;
				  fk2  =  a1*(vn+dt*fk1) + b1;
				  vnew[i] = vn + dt2*(fk1+fk2);
			  }
		  }
	  }
      return;
}
