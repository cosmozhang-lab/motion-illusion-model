// head segment
# include <fstream>
# include <string>
# include <iostream>
# include <sstream>
# include "math.h"
# include <float.h>
# include  <time.h>
# include <cstdlib>
# include <memory.h>
# include <cstdlib>
# define ni  64 
# define nj  64
# define nmax 4096
# define nlx 64 
# define nly 64 
# define nlgn 4096 
# define maxlgn 30 
# define nmap 6 
# define npars 64
# define twopi 6.28318
# define npat 18    
# define tncycle 25
using namespace std;


// drifting grating parameters(outside)
double omega    = 8.0; // temporal frequency
double gk       = 80.0;// spatial frequency 
double deltaDG  = twopi/npat;
int    myid     = 1;
double gtheta   = 0.0 + 1*3.14/4.0;				// orientation = pi
double gphi     = 0.00;

// location of LGN cells
/*----------location&number of LGN and V1 neurons------------*/
double *xlgn    = new double[nlgn];
double *ylgn    = new double[nlgn];
int    *nonlgn  = new int[nmax];
int    *noflgn  = new int[nmax];
int    *nlgni   = new int[nmax];
int    **ionlgn = new int*[nmax];
int    **ioflgn = new int*[nmax];

/*------------drift-grating stimuli cause fr of LGN-----------*/
double *ampcon = new double[nlgn];double *ampson = new double[nlgn];
double *ampcof = new double[nlgn];double *ampsof = new double[nlgn];
double *contr  = new double[200];


// spatial architecture of V1 neurons
double *phase    = new double[nmax];
double *theta    = new double[nmax];
int *clusthyp    = new int[nmax];
int *clustorien  = new int[nmax];

/*S/C cells*/
double *SCind    = new double[nmax];

// cortico-cortical connections
double *see    = new double[nmax];
double *sei    = new double[nmax];
double *sie    = new double[nmax];
double *sii    = new double[nmax];

//slow SR connections
double *seel    = new double[nmax];
double *siel    = new double[nmax];
//LR connections
// fast LR connections
double lee    = 0;
double lie    = 0;
// fast LR connections
double leef   = 0;
double lief   = 0;



// LGN conductance
// ON LGN fast/NMDA
double *glo    = new double[nlgn];double *slo3   = new double[nlgn];
double *glon   = new double[nlgn];double *slon3  = new double[nlgn];
// OFF LGN fast/NMDA
double *glf    = new double[nlgn];double *slf3   = new double[nlgn];
double *glfn   = new double[nlgn];double *slfn3  = new double[nlgn];
// external inhibitory units
double *gn     = new double[nmax];double *sn3    = new double[nmax];
double *gm     = new double[nmax];double *sm3    = new double[nmax];

/* adaptation terms
double *dlo    = new double[nmax];double *slo   = new double[nmax];
double *dlf    = new double[nmax];double *slf   = new double[nmax];
*/


// corresponding previous
//-----------previous----------------------------------------
double *glop   = new double[nlgn];double *slo3p  = new double[nlgn];
double *glonp  = new double[nlgn];double *slon3p = new double[nlgn];
double *glfp   = new double[nlgn];double *slf3p  = new double[nlgn];
double *glfnp  = new double[nlgn];double *slfn3p = new double[nlgn];
/*
double *dlop    = new double[nmax];double *slop   = new double[nmax];
double *dlfp    = new double[nmax];double *slfp   = new double[nmax];
*/

// effective lgn-connected
double *nlgneff= new double[nmax];
// previous spiking time of LGN cells and Inhibitory unit
double *pspon    = new double[nlgn];double *pspof   = new double[nlgn];
double *pspinh   = new double[nmax];
double cond0    ;



//V1 neurons connections
double *gl     = new double[nmax];
double *gi     = new double[nmax];double *si3    = new double[nmax]; // inibitory GABAA
double *gj     = new double[nmax];double *sj3    = new double[nmax]; // inhibitory GABAB
double *ge     = new double[nmax];double *se3    = new double[nmax]; // excitatory AMPA
double *gf     = new double[nmax];double *sf3    = new double[nmax]; // excitatory NMDA
double *gx     = new double[nmax];double *sx3    = new double[nmax]; // excitatory noise AMPA
double *gy     = new double[nmax];double *sy3    = new double[nmax]; // excitatory noise NMDA
// LR connections(only excitatory)
double *gel    = new double[nmax];double *sel3   = new double[nmax]; // excitatory AMPA LR
double *gfl    = new double[nmax];double *sfl3   = new double[nmax]; // excitatory NMDA LR


// coresponding previous
//-----------previous----------------------------------------
double *gip    = new double[nmax];double *si3p   = new double[nmax];
double *gjp    = new double[nmax];double *sj3p   = new double[nmax];
double *gep    = new double[nmax];double *se3p   = new double[nmax];
double *gfp    = new double[nmax];double *sf3p   = new double[nmax];
double *gxp    = new double[nmax];double *sx3p   = new double[nmax];
double *gyp    = new double[nmax];double *sy3p   = new double[nmax];
double *gmp    = new double[nmax];double *sm3p   = new double[nmax];
// LR connections(only excitatory)
double *gelp   = new double[nmax];double *sel3p  = new double[nmax]; // excitatory AMPA LR
double *gflp   = new double[nmax];double *sfl3p  = new double[nmax]; // excitatory NMDA LR


// recording conductance data
double **glgn   = new double*[nmax]; 
double **gexc   = new double*[nmax];double **ginh   = new double*[nmax];
double **gtot   = new double*[nmax];
double **cond   = new double*[nmax];double **vslave = new double*[nmax];
double **gnexc  = new double*[nmax];double **gninh  = new double*[nmax];
double **vmem   = new double*[nmax];

/*double **dplo   = new double*[nmax];
double **dplf   = new double*[nmax];*/
//-----------square-------------------------------------------
double **glgn2  = new double*[nmax]; 
double **gexc2  = new double*[nmax];double **ginh2  = new double*[nmax];
double **gtot2  = new double*[nmax];
double **cond2  = new double*[nmax];double **vslave2= new double*[nmax];
double **gnexc2 = new double*[nmax];double **gninh2 = new double*[nmax];
double **vmem2  = new double*[nmax];

// percent for GABAA,NMDA IN SR,NMDA IN TH,NMDA IN LR
double fgaba,fnmdac,fnmdat,fnmdalc;

// temporal parameters
double tau_e   = 0.001  ;double tnrise = 0.002; double tndamp = 0.08; 
double tau_i   = 0.00167;double tau2   = 0.007; double tstart = 0.25;
double frtlgn ;							// visual(frtlgn) void visual(double frtlgn0...) value transmitted and wouldn't be changed in function
// for adaptation
/*double tau_d   = 0.3    ;double tau_s  = 20;
double drd     = 0.75    ;double drs    = 0.99;*/

// connectivity matrix
double **icnntvy= new double*[nmax];
int    *indmap  = new int[nmax];
// Gaussian SR connections
double *a_ee   = new double[nmax];
double *a_ei   = new double[nmax];
double *a_ie   = new double[nmax];
double *a_ii   = new double[nmax];

// Gaussian LR connections
double *lr_ee   = new double[nmax];
double *lr_ie   = new double[nmax];


/*------------modified second order Runge-Kutta---------------*/
double *alpha0 = new double[nmax];double *beta0  = new double[nmax];
double *alpha1 = new double[nmax];double *beta1  = new double[nmax];
double *v      = new double[nmax];double *vnew   = new double[nmax];
double *pspike = new double[nmax];
// I&F voltage parameters
double vthres,tref,vreset;
double vexcit,vinhib,gleak;
// middle parameters
double ginhib,gexcit,ginoise,genoise;


// Output FR
/*------------V1 firing rate---------------------------------*/
int    **irate = new int*[nmax]; 
int **irateInh = new int*[nmax];
int    *ispike = new int[nmax] ;   double *tspike = new double[nmax];
int     nspike = 0             ;  
int     ist4   = 0             ;
int     nc     = 0             ;


// simulation parameters
double dt       = 0.0001  ;
double tfinal   = 5.0     ; double tstep1   = 1.0   ; double tstep2 = 1.0;
double tstep3   = 1.0     ; double tstep4   = 1.0;
unsigned iseed0 = 22594       ; unsigned iseed1 = 51967;
double ntotal   = tfinal/dt;
int    nstep1   = tstep1/dt;
int    nstep2   = tstep2/dt;
int    nstep3   = tstep3/dt;
int    nstep4   = tstep4/dt;
double period   = 1.0/omega;
int    nstep0   = (int)(period/dt/25.0);			// 1 T --> 25 bins
int    nsptot   = 0;int    isptot    = 0;



// function claim
void genmap(double,unsigned);
void gaussk(double ,double *,double *,int ,int ,double ,double ,double ,double );
void lgnrf(double );
void lgnsatur();
void e_or_i(unsigned);
void visual(double ,double ,double,double,double,double,unsigned);
void chain2(double *,double *,double ,double ,double ,int );
void chain(double *,double *,double ,double ,int );
void conduc(double *,double *,double ,double ,int);
void rk2_lif(double ,double );
void update(double );
void disort();
double newmod(double ,double );
void Preprocess(int);
/*void Drifting_Grating_Generator(int );
void Depression(double *,double ,double ,int );*/











