// all the functions
// function to calculate amp
void lgnrf(double frtlgn){
/*!------------------------------------------------------------
!  Map each LGN cell to its receptive field :
!	location & spatiotemporal filter parameters
!
!  Determine each cell's firing rate given visual stimulus :
!
!  	f(t)   = f0 + \int_0^t ds \int dy G(t - s) A(x - y) I(y,s)
!   int: integrate from 0 to t
!	  firing rate at time t for LGN cell centered at x
!
!	where
!
!	f0     = background rate
!       G(t)   = t^5[exp(-t/t0)/t0^6 - exp(-t/t1)/t1^6]
!	A(y)   = ampa/siga/pi exp(-y^2/siga^2) 
!		- ampb/sigb/pi exp(-y^2/sigb^2) 
!	 	(overall minus one factor for off-center LGN cells)
!	I(y,s) = I_0 [ 1 + eps sin(omega s) cos( k y - phi )
!		(for contrast reversal)
!----------------------------------------------------------
!------------------------------------------------------------
!     xlgn, ylgn: center of each LGN RF
!     siga, sigb: sigma of each Gaussian (RF = diff of 2 Gaussians)
!     ampa, ampb: amplitude of each Gaussian 
!     t0,   t1  : LGN kernel time constants
!     frtlgn    : background firing rate
!
!     9 Jan 2001: specialize to DG/CR for t >> t0
!	f(t) is a rectified sinusoid
!------------------------------------------------------------
/*------------------------------------------------------------
!  Hardwire each LGN cell
!------------------------------------------------------------*/
    double gkx    = gk*cos(gtheta);
    double gky    = gk*sin(gtheta);/* phase information and spatio-frequency*/

	  double contrast = 1.0;
	  double tau0 = 0.003;
	  double tau1 = 0.005;
	  double ot0  = omega*tau0;
	  double ot1  = omega*tau1;
	  double den0 = pow((1+ot0*ot0),6);
	  double den1 = pow((1+ot1*ot1),6);

	  double ampc = 240.0*((3 - 10*pow(ot0,2)+ 3*pow(ot0,4))*ot0 /
		  den0 - (3 - 10*pow(ot1,2) + 3*pow(ot1,4))*ot1 /
		  den1);
	  double amps = 120.0*((-1 + 15*pow(ot0,2) - 15*pow(ot0,4) + pow(ot0,6))
		  * ot0 / den0 
		  - (-1 + 15*pow(ot1,2) - 15*pow(ot1,4) + pow(ot1,6)) 
		  * ot1 / den1);
	  double tmpphi = atan2(amps,ampc);
	  double gk2    = gkx*gkx + gky*gky;
	  double gk2opt = 10.0*10.0;
	  double siga2  = 1.0/40.0/40.0;;
	  double sigb2  = 1.5*1.5 * siga2;
	  double dogopt = exp(-gk2opt*siga2/4.0) - 0.84*exp(-gk2opt*sigb2/4.0);
	  double doggk2 = exp(-gk2   *siga2/4.0) - 0.84*exp(-gk2   *sigb2/4.0);
	//------------------------------------------------------------
	  double ampccc = ampc/cos(tmpphi) * frtlgn/33.56*14.0;
	  ampccc = ampccc * doggk2/dogopt * contrast;
	 // printf("ampccc = %lf\n",ampccc);
	//------------------------------------------------------------
	  for( int i = 0; i < nlgn; i++){
		  ampson[i]  = -ampccc*cos(gphi-gkx*xlgn[i]-gky*ylgn[i]);
		  ampsof[i]  = -ampson[i];
		  ampcon[i]  =  ampccc*sin(gphi-gkx*xlgn[i]-gky*ylgn[i]);
		  ampcof[i]  = -ampcon[i];
	  }
	//------------------------------------------------------------
	  return ;
}