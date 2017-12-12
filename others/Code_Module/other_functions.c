// other functions
void disort(){
	double *tspikeb = new double[nmax]; int *ispikeb = new int[nmax];
	int    *order   = new int   [nmax];
	int    tempord  = 0;				double temp  = 0.0;
	for(int i = 0; i < nspike;i++){
		tspikeb[i] = tspike[i];
		ispikeb[i] = ispike[i];
	}
	for(int i = 0;i < nspike-1; i++){
		for(int j = 0; j < nspike-1-i;j++){
			if(tspike[j]>tspike[j+1]){
				temp        = tspike[j];
				tspike[j]   = tspike[j+1];
				tspike[j+1] = temp;
				tempord     = ispike[j];
				ispike[j]   = ispike[j+1];
				ispike[j+1] = tempord;
			}
		}
	}
	return ;
}


double newmod(double t,double tperiod){
	double tmp  = t/tperiod;
	int    ztmp = (int)tmp;
	double re   = (tmp-ztmp)*tperiod;
	return re;
}


void Drifting_Grating_Generator(int flag){
	if(flag ==0){
		FILE *SO,*SF,*CO,*CF;
		if((SO=fopen("ampson.txt", "rb")) == NULL) {
			printf("请确认文件(%s)是否存在!\n", "ampson.txt");
			exit(1);
		}else{
			for(int j = 0; j < nlgn; j++){
				fscanf(SO,"%lf ",&ampson[j]);
			}
		}

		if((SF=fopen("ampsof.txt", "rb")) == NULL) {
			printf("请确认文件(%s)是否存在!\n", "ampsof.txt");
			exit(1);
		}else{
			for(int j = 0; j < nlgn; j++){
				fscanf(SF,"%lf ",&ampsof[j]);
			}
		}

		if((CO=fopen("ampcon.txt", "rb")) == NULL) {
			printf("请确认文件(%s)是否存在!\n", "ampcon.txt");
			exit(1);
		}else{
			for(int j = 0; j < nlgn; j++){
				fscanf(CO,"%lf ",&ampcon[j]);
			}
		}

		if((CF=fopen("ampcof.txt", "rb")) == NULL) {
			printf("请确认文件(%s)是否存在!\n", "ampcof.txt");
			exit(1);
		}else{
			for(int j = 0; j < nlgn; j++){
				fscanf(CF,"%lf ",&ampcof[j]);
			}
		}
		fclose(SO);fclose(CO);fclose(SF);fclose(CF);
	}else{
		FILE *SO,*SF,*CO,*CF;
		printf("changing orientation of drifting grating .\n");

		if((SO=fopen("ampson1.txt", "rb")) == NULL) {
			printf("请确认文件(%s)是否存在!\n", "ampson.txt");
			exit(1);
		}else{
			for(int j = 0; j < nlgn; j++){
				fscanf(SO,"%lf ",&ampson[j]);
			}
		}

		if((SF=fopen("ampsof1.txt", "rb")) == NULL) {
			printf("请确认文件(%s)是否存在!\n", "ampsof.txt");
			exit(1);
		}else{
			for(int j = 0; j < nlgn; j++){
				fscanf(SF,"%lf ",&ampsof[j]);
			}
		}

		if((CO=fopen("ampcon1.txt", "rb")) == NULL) {
			printf("请确认文件(%s)是否存在!\n", "ampcon.txt");
			exit(1);
		}else{
			for(int j = 0; j < nlgn; j++){
				fscanf(CO,"%lf ",&ampcon[j]);
			}
		}

		if((CF=fopen("ampcof1.txt", "rb")) == NULL) {
			printf("请确认文件(%s)是否存在!\n", "ampcof.txt");
			exit(1);
		}else{
			for(int j = 0; j < nlgn; j++){
				fscanf(CF,"%lf ",&ampcof[j]);
			}
		}
		fclose(SO);fclose(CO);fclose(SF);fclose(CF);
	}
}


void Depression(double *d,double tau,double deltat,int nn){
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
	  
	  for(int i = 0; i < nn; i++){
		  *(d+i)  =  *(d+i) * ete + (1- ete);
	  }
      return;
}