// main algorithm after configuration
/*------------------------------------------------------------
 !  Grating parameters: Drift frequency, spatial frequency
 !------------------------------------------------------------*/
    omega  = twopi*omega;
    double gkx    = gk*cos(gtheta);
    double gky    = gk*sin(gtheta);
    //auto generate amp...
 	  lgnrf(frtlgn);
    e_or_i(iseed0);
    
    // display Exc vs. Inh counts
    double count =  0.0;
	  for (int i = 0; i< nmax/2; i++){
		  count = count + exc[i];
	  }
    double c1 = count;
	  for (int i = nmax/2; i < nmax; i++){
		  count = count + exc[i];
	  }
    printf("No. of Exc/Inh Cells : %d  %d\n",(int)count,nmax-(int)count);
    printf("No. of E/I Cells in Left column : %d  %d\n",(int)c1,nmax/2-(int)c1);
    printf("No. of E/I Cells in Right column : %d  %d\n",(int)(count-c1),nmax/2-(int)(count-c1));
//-----------initiate----------------------------------------
	  for(int i = 0 ;i < nlgn;i++){
		  glop[i]  = 0.0;slo3p[i]  = 0.0;
		  glonp[i] = 0.0;slon3p[i] = 0.0;
		  glfp[i]  = 0.0;slf3p[i]  = 0.0;
		  glfnp[i] = 0.0;slfn3p[i] =0.0;
	  }
	  for(int i = 0;i <nmax; i++){
		  gnp[i]   = 0.0;sn3p[i]   = 0.0;
		  gmp[i]   = 0.0;sm3p[i]   = 0.0;
	  }
	  for (int i = 0 ;i < nlgn; i++){
		  pspon[i]  = -10000.0;
		  pspof[i]  = -10000.0;
	  }
//---------------------------------Total transient 0.1 seconds
    double t   = -0.1;
    int ntrans = (int)(fabs(t)/dt);
	  
    double frtinhE = 200;				// base firing rate for inhibitory noise unit(Exc)
	  double ciE0    = 0.1;	
	  double frtinhI = 200.0;
	  double ciI0    = 0.1;


	  for(int i = 0; i <nmax; i++){
		  if ((frtinhE>0.0)&&(ciE0>0.0)){
			  srand(iseed1);
			  pspinh[i] =  t - log(rand()/(RAND_MAX+1.0))/frtinhE;
		  }
	  }
	  srand(iseed1);
	  for(int i = 0; i < ntrans ; i++){
		  visual(frtlgn,frtinhE,ciE0,frtinhI,ciI0,t,iseed1);
		  chain(glo,slo3,tau_e,dt,nlgn);
		  chain2(glon,slon3,tnrise,tndamp,dt,nlgn);
		  chain(glf,slf3,tau_e,dt,nlgn);
		  chain2(glfn,slfn3,tnrise,tndamp,dt,nlgn);
		  chain(gn,sn3,tau_i,dt,nmax);
		  chain(gm,sm3,tau2,dt,nmax);
		  t = t + dt;
	  }
//------------Initiate Parameters for main time integration----
	  gleak   = 50.0;
	  genoise = 0.0 ; ginoise = 0.0; gexcit = 0.0;  ginhib = 0.0;
	  vthres  =1.0  ; vexcit  =4.67; vinhib = -0.67;vreset = 0.0; 
	  // prepare to record firing rate
    for(int j = 0;j < nmax; j++){
	    irate[j]   = new int();
    }

/*-------------open an out.txt to record streamout-----------*/
	  FILE *Vmem,*Vslave,*Glgn,*Gexc,*Ginh,*Gtot;
	  // voltage
	  if((Vmem = fopen("vmem.txt","w"))==NULL)
	  {
		  printf("cannot open the output file exactly!\n");
		  exit(0);
	  }
	  if((Vslave = fopen("vslave.txt","w"))==NULL)
	  {
		  printf("cannot open the output file exactly!\n");
		  exit(0);
	  }
		// conductance
	  if((Glgn = fopen("glgn.txt","w"))==NULL)
	  {
		  printf("cannot open the output file exactly!\n");
		  exit(0);
	  }
	  if((Gexc= fopen("gexc.txt","w"))==NULL)
	  {
		  printf("cannot open the output file exactly!\n");
		  exit(0);
	  }
	  if((Ginh = fopen("ginh.txt","w"))==NULL)
	  {
		  printf("cannot open the output file exactly!\n");
		  exit(0);
	  }
	  if((Gtot = fopen("gtot.txt","w"))==NULL)
	  {
		  printf("cannot open the output file exactly!\n");
		  exit(0);
	  }

//	  FILE *dponlgn,*dpoflgn;
//	  if((dponlgn = fopen("dponlgn.txt","w"))==NULL)
//	  {
//		  printf("cannot open the output file exactly!\n");
//		  exit(0);
//	  }
//
//	  if((dpoflgn = fopen("dpoflgn.txt","w"))==NULL)
//	  {
//		  printf("cannot open the output file exactly!\n");
//		  exit(0);
//	  }


//-------------Start of Main Time Integration Loop for each dt
	  for ( int iii = 0; iii < ntotal; iii++){
	  	// RK2 algorithm
		  rk2_lif(t,dt);
		  // visual stimulus input
		  visual(frtlgn,frtinhE,ciE0,frtinhI,ciI0,t,iseed1);
		  // update conductance information
		  update(t);
//-------------------------finally advance TIME
		  t = t + dt;
		  for(int i = 0;i < nmax; i++){			  
			  v[i] = vnew[i];
			  if(vnew[i]<0.0){
				  v[i] = 0.0;
			  }
		  }
//-------------------------Cycle Conductances Summation
		  //					       output file f4
		  if (((iii% nstep0)==0)&&(t > 1.0)){//(t > 2.0
			  double tcycle,gei,gii,vsi;
			  int    ncycle;
			  // calculate the exact time bin(within a period)
			  tcycle = newmod(t-dt,period);
			  ncycle = (int)(tcycle*25.0/period);
			  for(int i = 0; i < nmax; i++){
				  if ( excite[i] ){
					  gei = ((1-fnmdac)*ge[i] + fnmdac*gf[i]) * see[i]+((1-fnmdalc)*gel[i] + fnmdalc*gfl[i]) * seel[i];
					  gii = ((1-fgaba)*gi[i] + fgaba*gj[i]) * sei[i];
				  }else{
					  gei = ((1-fnmdac)*ge[i] + fnmdac*gf[i]) * sie[i]+((1-fnmdalc)*gel[i] + fnmdalc*gfl[i]) * siel[i];
					  gii = ((1-fgaba)*gi[i] + fgaba*gj[i]) * sii[i];
				  }			  
				  vsi = -beta1[i]/alpha1[i];
				  
          glgn[i][ncycle] = gl[i] + glgn[i][ncycle];
				  gexc[i][ncycle] = gei + gexc[i][ncycle];
				  ginh[i][ncycle] = gii + ginh[i][ncycle];
				  gtot[i][ncycle] = -alpha1[i] + gtot[i][ncycle];
				  cond[i][ncycle] = beta1[i] + cond[i][ncycle];
				  vslave[i][ncycle] = vsi + vslave[i][ncycle];
				  vmem[i][ncycle]  = v[i] + vmem[i][ncycle];

//				  dplo[i][ncycle] = dlo[i] + dplo[i][ncycle];
//				  dplf[i][ncycle] = dlf[i] + dplf[i][ncycle];
			  }
		  }

//--------------------Cycle Average Conductance---------------
//--------------------nstep4 = Averaging Number * nstep0(1 period)---
		  if (((iii % nstep4)==0)&&(t > 1.0)){//(t > 2.0)){
			  ist4   = ist4 + 1;
			  nc     = nc + (int)(tstep4/period);
			  printf("The %d times averaging  \n",ist4);
			  for(int j = 0; j < tncycle; j++){			// 25 bins to be averaged
				  for(int i = 0; i < nmax; i++){
					  glgn[i][j]   = glgn[i][j]/nc;
					  gexc[i][j]   = gexc[i][j]/nc;
					  ginh[i][j]   = ginh[i][j]/nc;
					  gtot[i][j]   = gtot[i][j]/nc;
					  cond[i][j]   = cond[i][j]/nc;
					  vslave[i][j] = vslave[i][j]/nc;
					  vmem[i][j]   = vmem[i][j]/nc;
					  
					  fprintf(Vmem,"%lf ",vmem[i][j]);
				    fprintf(Vslave,"%lf ",vsi);
				    fprintf(Glgn,"%lf ",gl[i]);
				    fprintf(Gexc,"%lf ",gei);
				    fprintf(Ginh,"%lf ",gii);
				    fprintf(Gtot,"%lf ",gtot);

					  glgn[i][j]   = glgn[i][j]*nc;
					  gexc[i][j]   = gexc[i][j]*nc;
					  ginh[i][j]   = ginh[i][j]*nc;
					  gtot[i][j]   = gtot[i][j]*nc;
					  cond[i][j]   = cond[i][j]*nc;
					  vslave[i][j] = vslave[i][j]*nc;
					  vmem[i][j]   = vmem[i][j]*nc;

				  }
			  }
			  fprintf(Vmem,"\n ");
				fprintf(Vslave,"\n ");
				fprintf(Glgn,"\n ");
				fprintf(Gexc,"\n ");
				fprintf(Ginh,"\n ");
				fprintf(Gtot,"\n ");
		  }
}
}
