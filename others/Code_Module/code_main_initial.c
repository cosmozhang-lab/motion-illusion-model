// initiating module

// LGN input
// initiate 1D variables
memset(nlgni,0,nmax*sizeof(int));
memset(nonlgn,0,nmax*sizeof(int));
memset(noflgn,0,nmax*sizeof(int));
memset(xlgn,0,nlgn*sizeof(double));
memset(ylgn,0,nlgn*sizeof(double));
	
memset(ampson,0,nlgn*sizeof(double));
memset(ampsof,0,nlgn*sizeof(double));
memset(ampcon,0,nlgn*sizeof(double));
memset(ampcof,0,nlgn*sizeof(double));

// LGN conductance
memset(glo,0,nlgn*sizeof(double))  ;memset(slo3,0,nlgn*sizeof(double));
memset(glon,0,nlgn*sizeof(double)) ;memset(slon3,0,nlgn*sizeof(double));
memset(glf,0,nlgn*sizeof(double))  ;memset(slf3,0,nlgn*sizeof(double));
memset(glfn,0,nlgn*sizeof(double)) ;memset(slfn3,0,nlgn*sizeof(double));
memset(gn,0,nmax*sizeof(double))   ;memset(sn3,0,nmax*sizeof(double));
memset(gm,0,nmax*sizeof(double))   ;memset(sm3,0,nmax*sizeof(double));
// previous
memset(glop,0,nlgn*sizeof(double)) ;memset(slo3p,0,nlgn*sizeof(double));
memset(glonp,0,nlgn*sizeof(double));memset(slon3p,0,nlgn*sizeof(double));
memset(glfp,0,nlgn*sizeof(double)) ;memset(slf3p,0,nlgn*sizeof(double));
memset(glfnp,0,nlgn*sizeof(double));memset(slfn3p,0,nlgn*sizeof(double));
memset(gnp,0,nmax*sizeof(double))  ;memset(sn3p,0,nmax*sizeof(double));
memset(gmp,0,nmax*sizeof(double))  ;memset(sm3p,0,nmax*sizeof(double));


// initiate 2D variables
for(int i = 0; i < nmax; i++){
	ionlgn[i]  = new int[nlgn];memset(ionlgn[i],0,nlgn*sizeof(int));
	ioflgn[i]  = new int[nlgn];memset(ioflgn[i],0,nlgn*sizeof(int));
	nlgneff[i] = 20;
	}
memset(theta,0,nmax*sizeof(double));
memset(phase,0,nmax*sizeof(double));

memset(clustorien,0,nmax*sizeof(int));
memset(clusthyp,0,nmax*sizeof(int));

// V1 variables
// 1D V1 variables

// connections
memset(a_ee,0,nmax*sizeof(double));
memset(a_ei,0,nmax*sizeof(double));
memset(a_ie,0,nmax*sizeof(double));
memset(a_ii,0,nmax*sizeof(double));

memset(lr_ee,0,nmax*sizeof(double));
memset(lr_ie,0,nmax*sizeof(double));


memset(gl,0,nmax*sizeof(double))   ;
memset(gi,0,nmax*sizeof(double))   ;memset(si3,0,nmax*sizeof(double));printf("%lf",si3[0]);
memset(gj,0,nmax*sizeof(double))   ;memset(sj3,0,nmax*sizeof(double));
memset(ge,0,nmax*sizeof(double))   ;memset(se3,0,nmax*sizeof(double));
memset(gf,0,nmax*sizeof(double))   ;memset(sf3,0,nmax*sizeof(double));
memset(gx,0,nmax*sizeof(double))   ;memset(sx3,0,nmax*sizeof(double));
memset(gy,0,nmax*sizeof(double))   ;memset(sy3,0,nmax*sizeof(double));
// LR connections
memset(gel,0,nmax*sizeof(double))   ;memset(sel3,0,nmax*sizeof(double));
memset(gfl,0,nmax*sizeof(double))   ;memset(sfl3,0,nmax*sizeof(double));
//----------------previous-----------------------------------
memset(gip,0,nmax*sizeof(double))  ;memset(si3p,0,nmax*sizeof(double));
memset(gjp,0,nmax*sizeof(double))  ;memset(sj3p,0,nmax*sizeof(double));
memset(gep,0,nmax*sizeof(double))  ;memset(se3p,0,nmax*sizeof(double));
memset(gfp,0,nmax*sizeof(double))  ;memset(sf3p,0,nmax*sizeof(double));
memset(gxp,0,nmax*sizeof(double))  ;memset(sx3p,0,nmax*sizeof(double));
memset(gyp,0,nmax*sizeof(double))  ;memset(sy3p,0,nmax*sizeof(double));
// LR connections
memset(gelp,0,nmax*sizeof(double))  ;memset(sel3p,0,nmax*sizeof(double));
memset(gflp,0,nmax*sizeof(double))  ;memset(sfl3p,0,nmax*sizeof(double));

// sparse connectivity matrix
for(int i = 0; i < nmax; i++){
  icnntvy[i] = new double[nmap];
  memset(icnntvy[i],0,nmap*sizeof(double));
  //printf("icnn: %lf\n",icnntvy[0][3]);
}


// RK2 variables
memset(alpha0,0,nmax*sizeof(double))  ;memset(beta0,0,nmax*sizeof(double));
memset(alpha1,0,nmax*sizeof(double))  ;memset(beta1,0,nmax*sizeof(double));
memset(v,0,nmax*sizeof(double))       ;memset(vnew,0,nmax*sizeof(double));
memset(pspike,0,nmax*sizeof(double))  ;
memset(indmap,0,nmax*sizeof(int))     ;
// recording datas
for(int i = 0; i < nmax; i++){
	  glgn[i]    = new double[tncycle];memset(glgn[i],0,tncycle*sizeof(double));
	  gtot[i]    = new double[tncycle];memset(gtot[i],0,tncycle);
	  gexc[i]    = new double[tncycle];memset(gexc[i],0,tncycle*sizeof(double));
	  ginh[i]    = new double[tncycle];memset(ginh[i],0,tncycle*sizeof(double));
	  gnexc[i]   = new double[tncycle];memset(gnexc[i],0,tncycle*sizeof(double));
	  gninh[i]   = new double[tncycle];memset(gnexc[i],0,tncycle*sizeof(double));
	  cond[i]    = new double[tncycle];memset(cond[i],0,tncycle*sizeof(double));
	  vslave[i]  = new double[tncycle];memset(vslave[i],0,tncycle*sizeof(double));
	  vmem[i]    = new double[tncycle];memset(vmem[i],0,tncycle*sizeof(double));
	  glgn2[i]   = new double[tncycle];memset(glgn2[i],0,tncycle*sizeof(double));
	  gtot2[i]   = new double[tncycle];memset(gtot2[i],0,tncycle*sizeof(double));
	  gexc2[i]   = new double[tncycle];memset(gexc2[i],0,tncycle*sizeof(double));
	  ginh2[i]   = new double[tncycle];memset(ginh2[i],0,tncycle*sizeof(double));
	  gnexc2[i]  = new double[tncycle];memset(gnexc2[i],0,tncycle*sizeof(double));
	  gninh2[i]  = new double[tncycle];memset(gninh2[i],0,tncycle*sizeof(double));
	  cond2[i]   = new double[tncycle];memset(cond2[i],0,tncycle*sizeof(double));
	  vslave2[i] = new double[tncycle];memset(vslave2[i],0,tncycle*sizeof(double));
	  vmem2[i]   = new double[tncycle];memset(vmem2[i],0,tncycle*sizeof(double));
}

/*
memset(dlo,0,nmax*sizeof(double))  ;memset(slo,0,nmax*sizeof(double));
memset(dlf,0,nmax*sizeof(double))  ;memset(slf,0,nmax*sizeof(double));
memset(dlop,0,nmax*sizeof(double)) ;memset(slop,0,nmax*sizeof(double));
memset(dlfp,0,nmax*sizeof(double)) ;memset(slfp,0,nmax*sizeof(double));
for(int i = 0;i<nmax;i++){
	  dlo[i]  = 1.0;slo[i]  = 1.0;
	  dlop[i] = 1.0;slop[i] = 1.0;
	  dlf[i]  = 1.0;slf[i]  = 1.0;
	  dlfp[i] = 1.0;slfp[i] = 1.0;
}
*/

// others(more detailed
double *aa     = new double[nmax]     ;memset(aa,0,nmax*sizeof(double));


// detailed numbers
// percents for different receptors
fnmdac = 0.25;
fnmdat = 0.00;
fgaba  = 0.25;
fnmdalc= 0.25;

// base firing rate
frtlgn = 20.0;


// connections !!! iso/orientated
// axon and dentrite
double denexc = 50.0;
double deninh = 50.0;
double axnexc = 200.0;
double axninh = 100.0;
// spreading
double alee2  =  (denexc*denexc + axnexc*axnexc)/1.0E6;
double alei2  =  (denexc*denexc + axninh*axninh)/1.0E6;
double alie2  =  (deninh*deninh + axnexc*axnexc)/1.0E6;
double alii2  =  (deninh*deninh + axninh*axninh)/1.0E6;

// LR spreading 
double llee2  = 1.5*1.5/1.0;
double llie2  = 1.5*1.5/1.0;

double dx     =  0.5/ni;
double dy     =  0.5/nj;
double dx2    =  dx*dx;
double dy2    =  dy*dy;
double dxdy   =  dx*dy;
double dpi    =  0.5*twopi;
double cst    =  dxdy/dpi/ni/nj;

// restricts
double dnlgnmax = 30.0;      // link to 30 LGN cells
double g0       = 20.0;
double pconn    = 0.174046875;		// sparseness
cond0           =  g0/frtlgn/tau_e/dnlgnmax;

// for purely simple cells
double seemax = 0.25;
double seimax = 2.0;
double siemax = 6.0;
double siimax = 2.0;
// slow SR connections
double seelmax= 0.25;
double sielmax= 6.0;

// purely complex cells
double ceemax = 6.70;
double ceimax = 1.30;
double ciemax = 6.70;
double ciimax = 1.30;
// slow
double ceelmax= 6.70;
double cielmax= 6.70;

// clarrified S/C neurons
// save S/C index
FILE *SCIND;
if((SCIND = fopen("scind.txt","w"))==NULL)
{
  printf("cannot open the output file exactly!\n");
  exit(0);
}
double fee,fei,fie,fii;						//S/C index
for (int i=0;i<nmax;i++){
	SCind[i] = nlgni[i]/dnlgnmax;
	fprintf(SCIND,"%lf ",SCind[i]);
	/*if(SCind[i]>0.4){
		see[i] = seemax;
		sei[i] = seimax;
		sie[i] = siemax;
		sii[i] = siimax;
		}else{
			see[i] = ceemax;
			sei[i] = ceimax;
			sie[i] = ciemax;
			sii[i] = ciimax;
		}*/
  // S/C ind
  fee = 1.0 + (ceemax-seemax)*(1.0-nlgni[i]/dnlgnmax)/seemax;
  fei = 1.0 + (ceimax-seimax)*(1.0-nlgni[i]/dnlgnmax)/seimax;
  fie = 1.0 + (ciemax-siemax)*(1.0-nlgni[i]/dnlgnmax)/siemax;
  fii = 1.0 + (ciimax-siimax)*(1.0-nlgni[i]/dnlgnmax)/siimax;
  
  feel= 1.0 + (ceelmax-seelmax)*(1.0-nlgni[i]/dnlgnmax)/seelmax;
  fiel= 1.0 + (cielmax-sielmax)*(1.0-nlgni[i]/dnlgnmax)/sielmax;

  see[i] = seemax * fee;
  sei[i] = seimax * fei;
  sie[i] = siemax * fie;
  sii[i] = siimax * fii;
  // slow SR connections
  seel[i]= seelmax*feel;
  siel[i]= sielmax*fiel;
  // modified using sparseness
  see[i] = see[i] / tau_e * 4.0/3.0  / pconn;
  sei[i] = sei[i] / tau_i * 4.0     / pconn ;
  sie[i] = sie[i] / tau_e * 4.0/3.0  / pconn;
  sii[i] = sii[i] / tau_i * 4.0     / pconn ;
  
  seel[i] = seel[i] / tau_e * 4.0/3.0  / pconn;
  siel[i] = siel[i] / tau_e * 4.0/3.0  / pconn;
}

// normalized LR connection
// initiate LR connections
lee = 0.1;
lie = 0.1;
leef= 0.1;
lief= 0.1; 

// percents for global conductance
double fgee = 0.0;
double fgie = 0.0;
double fgei = 0.0;
double fgii = 0.0;

double fglee= 0.0;
double fglei= 0.0;


// configuration
srand(iseed0);
// generate sparse connections
genmap(pconn,iseed0);
// SR iso-orientated connections
gaussk(1.0,a_ee,aa,ni,nj,alee2,dx2,dy2,fgee);
gaussk(1.0,a_ei,aa,ni,nj,alei2,dx2,dy2,fgei);
gaussk(1.0,a_ie,aa,ni,nj,alie2,dx2,dy2,fgie);
gaussk(1.0,a_ii,aa,ni,nj,alii2,dx2,dy2,fgii);
// LR orientation-specific connections
gaussk(1.0,lr_ee,aa,ni,nj,llee2,dx2,dy2,fglee);
gaussk(1.0,lr_ie,aa,ni,nj,llie2,dx2,dy2,fglie);



// loading .txt files
void Preprocess(){
//	FILE *SO,*SF,*CO,*CF;
//	if((SO=fopen("ampson.txt", "rb")) == NULL) {
//			printf("请确认文件(%s)是否存在!\n", "ampson.txt");
//			exit(1);
//		}else{
//			for(int j = 0; j < nlgn; j++){
//				fscanf(SO,"%lf ",&ampson[j]);
//			}
//		}
//
//		if((SF=fopen("ampsof.txt", "rb")) == NULL) {
//			printf("请确认文件(%s)是否存在!\n", "ampsof.txt");
//			exit(1);
//		}else{
//			for(int j = 0; j < nlgn; j++){
//				fscanf(SF,"%lf ",&ampsof[j]);
//			}
//		}
//
//		if((CO=fopen("ampcon.txt", "rb")) == NULL) {
//			printf("请确认文件(%s)是否存在!\n", "ampcon.txt");
//			exit(1);
//		}else{
//			for(int j = 0; j < nlgn; j++){
//				fscanf(CO,"%lf ",&ampcon[j]);
//			}
//		}
//
//		if((CF=fopen("ampcof.txt", "rb")) == NULL) {
//			printf("请确认文件(%s)是否存在!\n", "ampcof.txt");
//			exit(1);
//		}else{
//			for(int j = 0; j < nlgn; j++){
//				fscanf(CF,"%lf ",&ampcof[j]);
//			}
//		}
//		fclose(SO);fclose(CO);fclose(SF);fclose(CF);
  FILE *NONLGN,*IDON,*NOFLGN,*IDOF;
  // load numbers and then index
  if((NONLGN=fopen("nonlgn.txt", "rb")) == NULL) {
  	printf("请确认文件(%s)是否存在!\n", "nonlgn.txt");
  	exit(1);
  }else{
  	for(int j = 0; j < nmax; j++){
  		fscanf(NONLGN,"%d ",&nonlgn[j]);
  	}
  }
  if((IDON=fopen("ionlgn.txt", "rb")) == NULL) {
  	printf("请确认文件(%s)是否存在!\n", "ionlgn.txt");
  	exit(1);
  }else{
  	for(int j = 0; j < nmax; j++){
  		for(int i = 0; i< nonlgn[j];i++){
  			fscanf(IDON,"%d ",&ionlgn[j][i]);
  		}
  	}
  }
  fclose(NONLGN);fclose(IDON);
  
  if((NOFLGN=fopen("noflgn.txt", "rb")) == NULL) {
  	printf("请确认文件(%s)是否存在!\n", "noflgn.txt");
  	exit(1);
  }else{
  	for(int j = 0; j < nmax; j++){
  		fscanf(NOFLGN,"%d ",&noflgn[j]);
  	}
  }
  if((IDOF=fopen("ioflgn.txt", "rb")) == NULL) {
  	printf("请确认文件(%s)是否存在!\n", "ioflgn.txt");
  	exit(1);
  }else{
  	for(int j = 0; j < nmax; j++){
  		for(int i = 0; i< noflgn[j];i++){
  			fscanf(IDOF,"%d ",&ioflgn[j][i]);
  		}
  	}
  }
  fclose(NOFLGN);fclose(IDOF);
  for(int j = 0;j<nmax;j++){
  	nlgni[j] = nonlgn[j]+noflgn[j];
  }
  
  // load coordinates for LGN cells
  FILE *XLGN,*YLGN;
  if((XLGN=fopen("xlgn.txt", "rb")) == NULL) {
  	printf("请确认文件(%s)是否存在!\n", "xlgn.txt");
  	exit(1);
  }else{
  	for(int j = 0; j < nlgn; j++){
  		fscanf(XLGN,"%lf ",&xlgn[j]);
  	}
  }
  
  if((YLGN=fopen("ylgn.txt", "rb")) == NULL) {
  	printf("请确认文件(%s)是否存在!\n", "ylgn.txt");
  	exit(1);
  }else{
  	for(int j = 0; j < nlgn; j++){
  		fscanf(YLGN,"%lf ",&ylgn[j]);
  	}
  }
  fclose(XLGN);fclose(YLGN);
  
  // load preferred orientation and phase
  FILE *ORIEN,*PHA;
  if((ORIEN=fopen("theta.txt", "rb")) == NULL) {
  	printf("请确认文件(%s)是否存在!\n", "theta.txt");
  	exit(1);
  }else{
  	for(int j = 0; j < nmax; j++){
  		fscanf(ORIEN,"%lf ",&theta[j]);
  	}
  }
  
  if((PHA=fopen("phase.txt", "rb")) == NULL) {
  	printf("请确认文件(%s)是否存在!\n", "phase.txt");
  	exit(1);
  }else{
  	for(int j = 0; j < nmax; j++){
  		phase[j] = rand()/(RAND_MAX+1.0);
  	}
  }
  fclose(ORIEN);fclose(PHA);
  
  // load columns information
  FILE *hypcol,*oriencol;
  if((hypcol=fopen("clusthyp.txt", "rb")) == NULL) {
  	printf("请确认文件(%s)是否存在!\n", "clusthyp.txt");
  	exit(1);
  }else{
  	for(int j = 0; j < nmax; j++){
  		fscanf(hypcol,"%d ",&clusthyp[j]);
  	}
  }
  fclose(hypcol);
  
  if((oriencol=fopen("clustorien.txt", "rb")) == NULL) {
  	printf("请确认文件(%s)是否存在!\n", "clustorien.txt");
  	exit(1);
  }else{
  	for(int j = 0; j < nmax; j++){
  		fscanf(oriencol,"%d ",&clustorien[j]);
  	}
  }
  fclose(oriencol);
	  
}



/* OPEN WRITING FILE FOR PREPARATION
/*-------------open an out.txt to record streamout-----------*/
	  FILE *Vmem,*Vslave,*Glgn,*Gexc,*Ginh,*Gtot,*FR;
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
	  
	  if((FR = fopen("fr.txt","w"))==NULL)
	  {
		  printf("cannot open the output file exactly!\n");
		  exit(0);
	  }
