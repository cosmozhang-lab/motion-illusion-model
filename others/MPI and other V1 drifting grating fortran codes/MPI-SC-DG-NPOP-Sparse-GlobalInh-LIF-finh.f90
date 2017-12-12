!***********************************************************
!
! Sparse coupling LIF code
!
!  8/2005: MPI version (runs 8 dg & 1 spontaneous on 9 processors OR
!		16 dg & 1 spontaneous on 17 processors)
!
!  5/24 2 Pop Model, LGN density  16 x 16 in 1 mm^2
!    global inhibition, synaptic failure, variable pdrive
!
!  Integrate-and-Fire Model Cortex
!
!  Background Constant Rate
!
!************************************************************
      program DGLIF
      use mpi
!      Use the following include if the mpi module is not available
!      include "mpif.h"
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
      parameter ( nlx = 48 , nly = 64 , nlgn = nlx*nly , maxlgn = 60 )
      parameter ( nmap = 32 , npars = 64)
      dimension params(npars)
      dimension see(nmax),sei(nmax),sie(nmax),sii(nmax)
      dimension aa(nmax),gl(nmax)
      dimension v(nmax),vnew(nmax),alpha1(nmax),beta1(nmax)
      dimension a_ee(nmax),a_ei(nmax),a_ie(nmax),a_ii(nmax)
      dimension glo(nlgn),slo1(nlgn),slo2(nlgn),slo3(nlgn)
      dimension glon(nlgn),slon1(nlgn),slon2(nlgn),slon3(nlgn)
      dimension glf(nlgn),slf1(nlgn),slf2(nlgn),slf3(nlgn)
      dimension glfn(nlgn),slfn1(nlgn),slfn2(nlgn),slfn3(nlgn)
      dimension ge(nmax),se1(nmax),se2(nmax),se3(nmax)
      dimension gf(nmax),sf1(nmax),sf2(nmax),sf3(nmax)
      dimension gi(nmax),si1(nmax),si2(nmax),si3(nmax)
      dimension gj(nmax),sj1(nmax),sj2(nmax),sj3(nmax)
      dimension gx(nmax),sx1(nmax),sx2(nmax),sx3(nmax)
      dimension gy(nmax),sy1(nmax),sy2(nmax),sy3(nmax)
      dimension gn(nmax),sn1(nmax),sn2(nmax),sn3(nmax)
      dimension gm(nmax),sm1(nmax),sm2(nmax),sm3(nmax)
      dimension glop(nlgn),slo1p(nlgn),slo2p(nlgn),slo3p(nlgn)
      dimension glonp(nlgn),slon1p(nlgn),slon2p(nlgn),slon3p(nlgn)
      dimension glfp(nlgn),slf1p(nlgn),slf2p(nlgn),slf3p(nlgn)
      dimension glfnp(nlgn),slfn1p(nlgn),slfn2p(nlgn),slfn3p(nlgn)
      dimension gep(nmax),se1p(nmax),se2p(nmax),se3p(nmax)
      dimension gfp(nmax),sf1p(nmax),sf2p(nmax),sf3p(nmax)
      dimension gip(nmax),si1p(nmax),si2p(nmax),si3p(nmax)
      dimension gjp(nmax),sj1p(nmax),sj2p(nmax),sj3p(nmax)
      dimension gxp(nmax),sx1p(nmax),sx2p(nmax),sx3p(nmax)
      dimension gyp(nmax),sy1p(nmax),sy2p(nmax),sy3p(nmax)
      dimension gnp(nmax),sn1p(nmax),sn2p(nmax),sn3p(nmax)
      dimension gmp(nmax),sm1p(nmax),sm2p(nmax),sm3p(nmax)
      dimension xlgn(nlgn),ylgn(nlgn),contr(200)
      dimension nonlgn(nmax),noflgn(nmax),nlgni(nmax)
      dimension ionlgn(nmax,2*maxlgn),ioflgn(nmax,2*maxlgn)
      dimension pspon(nlgn),pspoff(nlgn),pspike(nmax)
      dimension pspexc(nmax),pspinh(nmax)
      dimension irate(nmax,25)
      dimension glgn(nmax,25),gexc(nmax,25),ginh(nmax,25),gtot(nmax,25)
      dimension cond(nmax,25),vslave(nmax,25)
      dimension glgn2(nmax,25),gexc2(nmax,25),ginh2(nmax,25)
      dimension gtot2(nmax,25),cond2(nmax,25),vslave2(nmax,25)
      dimension vmem(nmax,25),vmem2(nmax,25)
      dimension gnexc(nmax,25),gninh(nmax,25)
      dimension gnexc2(nmax,25),gninh2(nmax,25)
      dimension tspike(nmax),ispike(nmax)
      dimension exc(nmax)
      logical excite(nmax)
      dimension icnntvy(nmax,nmap),indmap(nmax)
      character*80 f0,f1,f2,f3,f4,fsps,finput,fn(18),flgn,fdat(18)
      character*80 f0a,f1a,f2a,f3a,f4a,fspsa,fnam
      real*4 fnsp
!------------------------------------------------------------
!  For each V1 cell ---
!   1) see, sei, sie, sii: intracortical coupling strengths
!   2) gl: summed LGN conductance (summing over all on & off LGN cells
!   3) nonlgn, noflgn: number of On and Off LGN cells sending afferents
!   4) ionlgn, ioflgn: index of On and Off LGN cells sending afferents
!
!  For each LGN cell
!   xlgn, ylgn: coordinates of the center of its "receptive field"
!------------------------------------------------------------
      common / smatrx / see,sei,sie,sii
      common / lgncnd / gl
      common / lgnmap / nonlgn,noflgn,ionlgn,ioflgn
      common / lgnpos / xlgn,ylgn
      common / sparse / icnntvy,indmap
!------------------------------------------------------------
!  glo, glon: On-centered LGN cells, AMPA & NMDA channels
!  glf, glfn: Off-centered LGN cells, AMPA & NMDA channels
!------------------------------------------------------------
      common / chaino / glo,slo1,slo2,slo3,glop,slo1p,slo2p,slo3p
      common / chaino2/ glon,slon1,slon2,slon3, &
		glonp,slon1p,slon2p,slon3p
      common / chainf / glf,slf1,slf2,slf3,glfp,slf1p,slf2p,slf3p
      common / chainf2/ glfn,slfn1,slfn2,slfn3,  &
		glfnp,slfn1p,slfn2p,slfn3p
!------------------------------------------------------------
!  ge, gf: intracortical excitation, AMPA & NMDA, resp.
!  gi, gj: intracortical inhibition, GABA_A & B, resp.
!------------------------------------------------------------
      common / chaine / ge,se1,se2,se3,gep,se1p,se2p,se3p
      common / chaine2/ gf,sf1,sf2,sf3,gfp,sf1p,sf2p,sf3p
      common / chaini / gi,si1,si2,si3,gip,si1p,si2p,si3p
      common / chainj / gj,sj1,sj2,sj3,gjp,sj1p,sj2p,sj3p
!------------------------------------------------------------
!  gx, gy: "background" excitation, AMPA & NMDA
!  gn, gm: "background" inhibition, GABA_A & B
!------------------------------------------------------------
      common / chainx / gx,sx1,sx2,sx3,gxp,sx1p,sx2p,sx3p
      common / chainy / gy,sy1,sy2,sy3,gyp,sy1p,sy2p,sy3p
      common / chainn / gn,sn1,sn2,sn3,gnp,sn1p,sn2p,sn3p
      common / chainm / gm,sm1,sm2,sm3,gmp,sm1p,sm2p,sm3p
!------------------------------------------------------------
!  Various reversal potentials and time constants
!------------------------------------------------------------
      common / vconst / vthres,vreset,vexcit,vinhib,gleak
      common / tconst / tau_e,tau_i,tau0,tau1,tau2,tnrise,tndamp
!------------------------------------------------------------
!  fnmdat(c) Fraction of NMDA for thalamocortical (& intracortical)
!  fgaba     Fraction of GABA_B
!------------------------------------------------------------
      common /  NMDA  / fnmdat,fnmdac,fgaba
!------------------------------------------------------------
!  a_xy: spatial kernels
!  excite:  = 1 for excitatory neurons, = 0 for inhibitory
!  nlgni:  No. of LGN afferents
!------------------------------------------------------------
      common / kernel / a_ee,a_ei,a_ie,a_ii
      common / neuron / excite,nlgni
!------------------------------------------------------------
!  Stimulus parameters:
!    omega: temporal frequency
!    (gkx,gky): grating direction vector
!    gphi: grating spatial phase
!
!    contr: array for LGN contrast saturation
!------------------------------------------------------------
      common / conrev / omega,gkx,gky,gphi,tstart,contrast
      common / consat / contr
      common /   rhs  / alpha1,beta1
      common /  avgs  / condavg,curravg,condlgn,condexc,condinh, &
		geavg,giavg
      common /   isi  / pspike,nsptot,isptot,nessp,necsp,nissp,nicsp
      common / spikes / irate,period
      common / lgnsps / pspon,pspoff
      common /  psps  / pspexc,pspinh
      common / ttotal / tfinal,twindow
      common / synapt / pconn
      common / filenm / fsps
      data iword / 8 /
      external chain
!------------------------------------------------------------
!  Some (not all) INPUT & OUTPUT file declarations
!------------------------------------------------------------
      iword2 = iword/2
      finput = 'INPUT'
      f0     = 'i-and-f.list'
      f1a    = 'i-and-f.dat1'
      f2a    = 'i-and-f.dat2'
      f3a    = 'i-and-f.dat3'
      f4a    = 'i-and-f.dat4'
      fspsa  = 'spikes.dat'
      flgn   = 'lgnmap.out'
      iseed0 =  22594
      fn(1)  = 'DGlif00/'
      fn(2)  = 'DGlif01/'
      fn(3)  = 'DGlif02/'
      fn(4)  = 'DGlif03/'
      fn(5)  = 'DGlif04/'
      fn(6)  = 'DGlif05/'
      fn(7)  = 'DGlif06/'
      fn(8)  = 'DGlif07/'
      fn(9)  = 'DGlif08/'
      fn(10) = 'DGlif09/'
      fn(11) = 'DGlif10/'
      fn(12) = 'DGlif11/'
      fn(13) = 'DGlif12/'
      fn(14) = 'DGlif13/'
      fn(15) = 'DGlif14/'
      fn(16) = 'DGlif15/'
      fn(17) = 'DGlif16/'
      fn(18) = 'DGlif17/'

      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)
!------------------------------------------------------------
! Read INPUT file for run parameters
!------------------------------------------------------------
      if (myid .eq. 0) then
	open(3,file=finput,access='sequential',form='formatted')
	read(3,*)
	read(3,*)
	read(3,*)
	read(3,*)
	read(3,*)
	read(3,*) dt,tfinal,tstep1,tstep2,tstep3,tstep4,iseed1
	read(3,*)
	read(3,*)
	read(3,*) vthres,vreset,vexcit,vinhib,gleak,fge,fgi
	read(3,*)
	read(3,*)
	read(3,*) tau_e,tau_i,tnrise,tndamp,tau2,pconn
	read(3,*)
	read(3,*)
	read(3,*) denexc,axnexc,deninh,axninh,fnmdat,fnmdac,fgaba
	read(3,*)
	read(3,*)
	read(3,*) frtlgn,g0,tau0,tau1,frtinhE,ciE0,frtinhI,ciI0
	read(3,*)
	read(3,*)
	read(3,*) omega,gk,gtheta,gphi,tstart,twindow,contrast
	read(3,*)
	read(3,*)
	read(3,*) seemax,seimax,siemax,siimax,ceemax,ceimax,ciemax,ciimax
	close(3)

	params(1) = dt
	params(2) = tfinal
        params(3) = tstep1
        params(4) = tstep2
        params(5) = tstep3
        params(6) = tstep4
        params(7) = vthres
        params(8) = vreset
        params(9) = vexcit
        params(10) = vinhib
        params(11) = gleak
        params(12) = fge
        params(13) = fgi
        params(14) = tau_e
        params(15) = tau_i
        params(16) = tnrise
        params(17) = tndamp
        params(18) = tau2
        params(19) = pconn
        params(20) = denexc
        params(21) = axnexc
        params(22) = deninh
        params(23) = axninh
        params(24) = fnmdat
        params(25) = fnmdac
        params(26) = fgaba
        params(27) = frtlgn
        params(28) = g0
        params(29) = tau0
        params(30) = tau1
        params(31) = frtinhE
        params(32) = ciE0
        params(33) = frtinhI
        params(34) = ciI0
        params(35) = omega
        params(36) = gk
        params(37) = gtheta
        params(38) = gphi
        params(39) = tstart
        params(40) = twindow
        params(41) = contrast
        params(42) = seemax
        params(43) = seimax
        params(44) = siemax
        params(45) = siimax
        params(46) = ceemax
        params(47) = ceimax
	params(48) = ciemax
        params(49) = ciimax
!        params(50) = vthres
!        params(51) = vt
!        params(52) = deltav
      endif

      call MPI_BCAST(params,npars,MPI_DOUBLE_PRECISION,0, &
		MPI_COMM_WORLD,ierr)
      call MPI_BCAST(iseed1,1,MPI_INTEGER4,0, &
		MPI_COMM_WORLD,ierr)

      if (myid .ne. 0 ) then
        dt     = params(1)
        tfinal = params(2)
        tstep1 = params(3)
        tstep2 = params(4)
        tstep3 = params(5)
        tstep4 = params(6)
        vthres = params(7)
        vreset = params(8)
        vexcit = params(9)
        vinhib = params(10)
        gleak  = params(11)
        fge    = params(12)
        fgi    = params(13)
        tau_e  = params(14)
        tau_i  = params(15)
        tnrise = params(16)
        tndamp = params(17)
        tau2   = params(18)
        pconn  = params(19)
        denexc = params(20)
        axnexc = params(21)
        deninh = params(22)
        axninh = params(23)
        fnmdat = params(24)
        fnmdac = params(25)
        fgaba  = params(26)
        frtlgn = params(27)
        g0     = params(28)
        tau0   = params(29)
        tau1   = params(30)
        frtinhE = params(31)
        ciE0    = params(32)
        frtinhI = params(33)
        ciI0    = params(34)
        omega  = params(35)
        gk     = params(36)
        gtheta = params(37)
        gphi   = params(38)
        tstart   = params(39)
        twindow  = params(40)
        contrast = params(41)
        seemax = params(42)
        seimax = params(43)
        siemax = params(44)
        siimax = params(45)
        ceemax = params(46)
        ceimax = params(47)
        ciemax = params(48)
        ciimax = params(49)
!        vthres = params(50)
!        vt     = params(51)
!        deltav = params(52)
!        tdd = 1.0d0*exp((vt-vthres)/deltav)/gleak
      endif

!      if (contrast .le. 0.0d0) then
	iseed = iseed1 + myid + 1
!      endif

      twopi  =  8.0d0*atan(1.0)
      deltaDG = twopi / 16.0d0
      gtheta = myid * deltaDG
!
! spontaneous run on 0th node
!
      if (myid .eq. 0) then
	gtheta = 0.0
	contrast = 0.0
      endif

      print *,'myid ',myid,' theta = ', sngl(gtheta), &
		sngl(contrast)

      twindow2 = 0.0d0
!------------------------------------------------------------
      if (tstart .lt. 0.1D0) tstart = 0.1D0
!------------------------------------------------------------
!     Read in LGN location and LGN-V1 map from file flgn
!------------------------------------------------------------
      if (myid .eq. 0) then
	open(14,file=flgn,access='sequential',form='formatted')
	print *,' Reading LGN locations'
	do i=1,nlgn
	  read(14,*) xlgn(i),ylgn(i)
	enddo
	print *,' Reading LGN-V1 map'
	do i=1,nmax
	  read(14,*) nonlgn(i),noflgn(i)
	  do j=1,nonlgn(i)
	    read(14,*) ionlgn(i,j)
	  enddo
	  do j=1,noflgn(i)
	    read(14,*) ioflgn(i,j)
	  enddo
	enddo
	close(14)
      endif

      call MPI_BCAST(xlgn,nlgn,MPI_DOUBLE_PRECISION,0, &
		MPI_COMM_WORLD,ierr)
      call MPI_BCAST(ylgn,nlgn,MPI_DOUBLE_PRECISION,0, &
		MPI_COMM_WORLD,ierr)
      call MPI_BCAST(nonlgn,nmax,MPI_INTEGER4,0, &
		MPI_COMM_WORLD,ierr)
      call MPI_BCAST(noflgn,nmax,MPI_INTEGER4,0, &
		MPI_COMM_WORLD,ierr)
      call MPI_BCAST(ionlgn,2*nmax*maxlgn,MPI_INTEGER4,0, &
		MPI_COMM_WORLD,ierr)
      call MPI_BCAST(ioflgn,2*nmax*maxlgn,MPI_INTEGER4,0, &
		MPI_COMM_WORLD,ierr)

      do i=1,nmax
	nlgni(i) = nonlgn(i) + noflgn(i)
      enddo
!------------------------------------------------------------
!     Initialize output files
!------------------------------------------------------------
      fnam = fn(myid+1)

      ind1 = index(fnam,' ')-1
      ind2 = index(fspsa,' ')-1
      fsps = fnam(1:ind1)//fspsa(1:ind2)
      open(20,file=fsps,status='new',form='unformatted', &
	access='direct',recl=iword2)
      fnsp = sngl(0.0d0)
      write(20,rec=1) fnsp
      close(20)

      ind1 = index(fnam,' ')-1
      ind2 = index(f1a,' ')-1
      f1   = fnam(1:ind1)//f1a(1:ind2)
      ist1 = 0
      open(11,file=f1,status='new',form='unformatted', &
	access='direct',recl=iword2*25*nmax)
      close(11)

      ind1 = index(fnam,' ')-1
      ind2 = index(f2a,' ')-1
      f2   = fnam(1:ind1)//f2a(1:ind2)
      ist2 = 0
      open(12,file=f2,status='new',form='unformatted', &
	access='direct',recl=iword*2*1001)
      close(12)

!      ind1 = index(fnam,' ')-1
!      ind2 = index(f3a,' ')-1
!      f3   = fnam(1:ind1)//f3a(1:ind2)
      ist3 = 0
      if (myid .eq. 0) then
	f3 = f3a
	open(13,file=f3,status='new',form='unformatted', &
		access='direct',recl=iword*nmax)
	close(13)
      endif

      ind1 = index(fnam,' ')-1
      ind2 = index(f4a,' ')-1
      f4   = fnam(1:ind1)//f4a(1:ind2)
      ist4 = 0
      nc   = 0
      open(14,file=f4,status='new',form='unformatted', &
	access='direct',recl=iword*nmax*6*25)
      close(14)

!      ncount = 0
!      do ii=1,100
!	nin(ii) = 0
!	open(7000+ii,status='new',form='unformatted',
!     1       access='direct',recl=iword*8)
!	close(7000+ii)
!      enddo
!------------------------------------------------------------
!     Initialize spatial constants and spatial kernels
!------------------------------------------------------------
      alee2  =  (denexc*denexc + axnexc*axnexc)/1.d6
      alei2  =  (denexc*denexc + axninh*axninh)/1.d6
      alie2  =  (deninh*deninh + axnexc*axnexc)/1.d6
      alii2  =  (deninh*deninh + axninh*axninh)/1.d6
      dx     =  1.d0/ni
      dy     =  1.d0/nj
      dx2    =  dx*dx
      dy2    =  dy*dy
      dxdy   =  dx*dy
      twopi  =  8.0d0*atan(1.0)
      dpi    =  0.5d0*twopi
      const  =  dxdy/dpi/ni/nj
!
!  cond0 is normalized to maximum g0 for n_LGN = 30
!
      dnlgnmax = 30.0d0
      cond0  =  g0/frtlgn/tau_e/dnlgnmax
      do i=1,nmax
!
! linear S between nlgn = 0 & nlgn = 30
!
	fee = 1.0D0 + (ceemax-seemax)*(1.0D0-nlgni(i)/dnlgnmax)/seemax
	fei = 1.0D0 + (ceimax-seimax)*(1.0D0-nlgni(i)/dnlgnmax)/seimax
	fie = 1.0D0 + (ciemax-siemax)*(1.0D0-nlgni(i)/dnlgnmax)/siemax
	fii = 1.0D0 + (ciimax-siimax)*(1.0D0-nlgni(i)/dnlgnmax)/siimax

	see(i) = seemax * fee
	sei(i) = seimax * fei
	sie(i) = siemax * fie
	sii(i) = siimax * fii
	see(i) = see(i) / tau_e * 4.D0/3.  / pconn
	sei(i) = sei(i) / tau_i * 4.D0     / pconn 
	sie(i) = sie(i) / tau_e * 4.D0/3.  / pconn
	sii(i) = sii(i) / tau_i * 4.D0     / pconn
      enddo
!------------------------------------------------------------
      fgee = 0.0d0
      fgie = 0.0d0
      fgei = fge
      fgii = fgi
!------------------------------------------------------------
      call genmap(icnntvy,indmap,pconn,iseed0,myid)
      call gaussk(1.D0,a_ee,aa,ni,nj,alee2,dx2,dy2,fgee)
      call gaussk(1.D0,a_ei,aa,ni,nj,alei2,dx2,dy2,fgei)
      call gaussk(1.D0,a_ie,aa,ni,nj,alie2,dx2,dy2,fgie)
      call gaussk(1.D0,a_ii,aa,ni,nj,alii2,dx2,dy2,fgii)

      if (myid .eq. 0) then
      print *,'--------------------------------------------'
      print *,'Integrate-and-Fire Network of ',ni,' x ',nj,' Neurons'
      print *,'--------------------------------------------'
      print *,'         leak  = ',sngl(gleak)
      print *,'       Vthres  = ',sngl(vthres),'     Ve = ',sngl(vexcit)
      print *,'       Vreset  = ',sngl(vreset),'     Vi = ',sngl(vinhib)
      print *,'--------------------------------------------'
      print *,'        spatial coupling : '
      print *,'        den_e  = ',sngl(denexc),'  den_i = ',sngl(deninh)
      print *,'        axn_e  = ',sngl(axnexc),'  axn_i = ',sngl(axninh)
      print *,'--------------------------------------------'
      print *,' synaptic time constants : '
      print *,'        tau_e  = ',sngl(tau_e)
      print *,'        tau_i  = ',sngl(tau_i), 'tau_i-2 = ',sngl(tau2)
      print *,'--------------------------------------------'
      print *,'  LGN-driving parameters : '
      print *,'  firing rate  = ',sngl(frtlgn),'     g0 = ',sngl(g0)
      print *,'--------------------------------------------'
      print *,'      synaptic strengths : '
      print *,'          See   = ',sngl(seemax), &
		'    Sie = ',sngl(siemax)
      print *,'          Sei   = ',sngl(seimax), &
		'    Sii = ',sngl(siimax)
      print *,'--------------------------------------------'
      print *,'        noise parameters : '
      print *,' poisson rate  = ',sngl(frtinhE),'    str = ',sngl(ciE0)
      print *,'      (inhib)  = ',sngl(frtinhI),'    str = ',sngl(ciI0)
      print *,'--------------------------------------------'
      print *,'      grating parameters : '
      print *,' spatial freq  = ',sngl(omega), '      k = ',sngl(gk)
      print *,'        angle  = ',sngl(gtheta),'  phase = ',sngl(gphi)
      print *,'       tstart  = ',sngl(tstart), &
		'twindow = ',sngl(twindow)
      print *,'--------------------------------------------'
      print *,'output and time-stepping : '
      print *,'       tfinal  = ',sngl(tfinal),'     dt = ',sngl(dt)
      print *,'       tstep1  = ',sngl(tstep1),' tstep2 = ',sngl(tstep2)
      print *,'       tstep3  = ',sngl(tstep3),' tstep4 = ',sngl(tstep4)
      print *,'--------------------------------------------'
      endif

      ntotal = tfinal/dt
      nstep1 = tstep1/dt
      nstep2 = tstep2/dt
      nstep3 = tstep3/dt
      nstep4 = tstep4/dt
      period = 1.0/omega
      nstep0 = period/dt/25.0
!------------------------------------------------------------
!  Grating parameters: Drift frequency, spatial frequency
!------------------------------------------------------------
      omega  = twopi*omega
      gkx    = gk*cos(gtheta)
      gky    = gk*sin(gtheta)
!------------------------------------------------------------
!  Input omega in Hertz
!----------------------Determine DRIFTING GRATING time course
!				   E/I location can be random
      call lgnrf(frtlgn)
      call lgnsatur
      call e_or_i(excite,exc,iseed0)
      if (myid .eq. 0) then
      open(13,file=f3,status='old',form='unformatted', &
	access='direct',recl=iword*nmax)
      write(13,rec=1) (see(i)*tau_e/4.D0*3.,i=1,nmax)
      write(13,rec=2) (sei(i)*tau_i/4.D0,i=1,nmax)
      write(13,rec=3) (sie(i)*tau_e/4.D0*3.,i=1,nmax)
      write(13,rec=4) (sii(i)*tau_i/4.D0,i=1,nmax)
      write(13,rec=5) exc
      write(13,rec=6) (nlgni(i)*1.0D0,i=1,nmax)
      close(13)
      count =  0.0D0
      do i=1,nmax/2
	count = count + exc(i)
      enddo
      c1 = count
      do i=nmax/2+1,nmax
	count = count + exc(i)
      enddo
      print *,'No. of Exc/Inh Cells : ',int(count),nmax-int(count)
      print *,'  No. of E/I Cells in Left column : ',int(c1), &
		nmax/2-int(c1)
      print *,'  No. of E/I Cells in Right column : ',int(count-c1), &
		nmax/2-int(count-c1)
      endif
!---------------------------------------Transient noise & lgn 
      do i=1,nlgn
	pspon(i)  = -10000.0
	pspoff(i) = -10000.0
      enddo
!---------------------------------Total transient 0.1 seconds
      t = -0.1D0
      ntrans = abs(t)/dt
      do i=1,nmax
!	if (frtexc.gt.0.0.and.ce0.gt.0.0D0) then
!	  pspexc(i) =  t - dlog(ran2(iseed))/frtexc
!	endif
	if (frtinhE.gt.0.0.and.ciE0.gt.0.0D0) then
	  pspinh(i) =  t - dlog(ran2(iseed))/frtinhE
	endif
      enddo

      do i=1,ntrans
	call visual(frtlgn,cond0,frtinhE,ciE0,frtinhI,ciI0,t,dt,iseed)
	call chain(glo,slo1,slo2,slo3,tau_e,dt,nlgn)
	call chain2(glon,slon1,slon2,slon3,tnrise,tndamp,dt,nlgn)
	call chain(glf,slf1,slf2,slf3,tau_e,dt,nlgn)
	call chain2(glfn,slfn1,slfn2,slfn3,tnrise,tndamp,dt,nlgn)
!	call chain(gx,sx1,sx2,sx3,tau_e,dt,nmax)
!	call chain2(gy,sy1,sy2,sy3,tnrise,tndamp,dt,nmax)
	call chain(gn,sn1,sn2,sn3,tau_i,dt,nmax)
	call chain(gm,sm1,sm2,sm3,tau2,dt,nmax)
	t = t + dt
      enddo
      if (myid .eq. 0) then
	print *,'After generating initial transients',t
      endif
      t      = 0.0d0

      nsptot = 0
      isptot = 0

      if (myid .eq. 0) then
      open(10,file=f0,status='new',form='formatted')
      write(10,9000)
      write(10,9000)
      write(10,9010)
      write(10,9000)
      write(10,9000)
      write(10,9030) ni,nj
      write(10,9040) vthres,vreset,vexcit,vinhib,tau_e,tau_i, &
	gleak,seemax,seimax,siemax,siimax, &
	int(denexc),int(axnexc),int(deninh),int(axninh)
      write(10,9000)
      write(10,9050) 
      write(10,9060) frtlgn,g0,tau0,tau1
      write(10,9000)
      write(10,9070) 
      write(10,9080) frtinhE,frtinhI,ciE0,ciI0
      write(10,9000)
      write(10,9100) ntotal,tfinal,nstep1,tstep1,nstep2,tstep2, &
	nstep3,tstep3,nstep4,tstep4
      write(10,9000)

      close(10)
      endif

      condavg = 0.D0
      curravg = 0.D0
      condlgn = 0.D0
      condinh = 0.D0
      condexc = 0.D0
      geavg   = 0.D0
      giavg   = 0.D0
      gmavg   = 0.D0
      gnavg   = 0.D0
      gxavg   = 0.D0
      gyavg   = 0.D0
      do i=1,nmax
	gl(i)	  = 0.0
	do j=1,nonlgn(i)
	  gl(i) = gl(i) + (1-fnmdat)*glo(ionlgn(i,j)) &
		+ fnmdat*glon(ionlgn(i,j))
	enddo
	do j=1,noflgn(i)
	  gl(i) = gl(i) + (1-fnmdat)*glf(ioflgn(i,j)) &
		+ fnmdat*glfn(ioflgn(i,j))
	enddo

	if ( excite(i) ) then
	  ginhib  = ((1-fgaba)*gi(i) + fgaba*gj(i))*sei(i)
	  gexcit  = ((1-fnmdac)*ge(i) + fnmdac*gf(i))*see(i)
	else
	  ginhib  = ((1-fgaba)*gi(i) + fgaba*gj(i))*sii(i)
	  gexcit  = ((1-fnmdac)*ge(i) + fnmdac*gf(i))*sie(i)
	endif

	ginoise   = (1-fgaba)*gn(i) + fgaba*gm(i)
	genoise   = (1-fnmdat)*gx(i) + fnmdat*gy(i)
	gmavg     =  gmavg + gm(i)
	gnavg     =  gnavg + gn(i)
	gxavg     =  gxavg + gx(i)
	gyavg     =  gyavg + gy(i)
	alpha1(i) = -gleak - gl(i) - gexcit - ginhib - ginoise - genoise
	beta1(i)  = (gl(i) + gexcit + genoise)*vexcit &
		+ (gnoise + ginhib)*vinhib
	condavg   =  condavg - alpha1(i)
	curravg   =  curravg + beta1(i)
	condlgn   =  condlgn + gl(i)
	condinh   =  condinh + ginoise
	condexc   =  condexc + genoise
	geavg     =  geavg   + gexcit
	giavg     =  giavg   + ginhib
      enddo
      if (myid .eq. 0) then
	print *,condavg/nmax,curravg/nmax,condlgn/nmax,condexc/nmax, &
		condinh/nmax,geavg/nmax,giavg/nmax

	print *,'Gm/n avg = ',gmavg/nmax,gnavg/nmax
	print *,'Gx/y avg = ',gxavg/nmax,gyavg/nmax

	print *
	print *,'      tauNMDA = ',sngl(tnrise),sngl(tndamp)
	print *,'  fNMDA tc/cc = ',sngl(fnmdat),sngl(fnmdac)
      endif
!-------------Start of Main Time Integration Loop for each dt
      do iii=1,ntotal
!------------------------------------------------------------
	call rk2_lif(chain,v,vnew,nspike,ispike,tspike,t,dt)
!!!	call rk2(chain,v,vnew,nspike,ispike,tspike,t,dt)
!!!	call rk4(chain,v,vnew,nspike,ispike,tspike,t,dt)
!-------------------------Generate noise and LGN input spikes 
	call visual(frtlgn,cond0,frtinhE,ciE0,frtinhI,ciI0,t,dt,iseed)
	call update(chain,nspike,ispike,tspike,dt,t,myid)
!----------------------------------------finally advance TIME
	t = t + dt
	do i=1,nmax
	  v(i) = vnew(i)
	enddo
!------------------------------------Cycle Avg'd Conductances
!					       output file f4
	if ((mod(iii,nstep0).eq.0).and.(t.gt.2.0d0)) then
!------------------------------------------------------------
	  tcycle = mod(t-dt,period)
	  ncycle = int(tcycle*25.D0/period) + 1
	  do i=1,nmax
	    if ( excite(i) ) then
	      gei = ((1-fnmdac)*ge(i) + fnmdac*gf(i)) * see(i)
	      gii = ((1-fgaba)*gi(i) + fgaba*gj(i)) * sei(i)
	    else
	      gei = ((1-fnmdac)*ge(i) + fnmdac*gf(i)) * sie(i)
	      gii = ((1-fgaba)*gi(i) + fgaba*gj(i)) * sii(i)
	    endif
!	    gen = (1-fnmda)*gx(i) + fnmda*gy(i)
!	    gin = 0.5D0 * (gm(i) + gn(i))

	    vsi = -beta1(i)/alpha1(i)
	    glgn(i,ncycle) = gl(i) + glgn(i,ncycle)
	    gexc(i,ncycle) = gei + gexc(i,ncycle)
	    ginh(i,ncycle) = gii + ginh(i,ncycle)
	    gtot(i,ncycle) = -alpha1(i) + gtot(i,ncycle)
	    cond(i,ncycle) = beta1(i) + cond(i,ncycle)
	    vslave(i,ncycle) = vsi + vslave(i,ncycle)
!	    gnexc(i,ncycle) = gen + gnexc(i,ncycle)
!	    gninh(i,ncycle) = gin + gninh(i,ncycle)

	    glgn2(i,ncycle) = gl(i)*gl(i) + glgn2(i,ncycle)
	    gexc2(i,ncycle) = gei*gei + gexc2(i,ncycle)
	    ginh2(i,ncycle) = gii*gii + ginh2(i,ncycle)
	    gtot2(i,ncycle) = alpha1(i)*alpha1(i) + gtot2(i,ncycle)
	    cond2(i,ncycle) = beta1(i)*beta1(i) + cond2(i,ncycle)
	    vslave2(i,ncycle) = vsi*vsi + vslave2(i,ncycle)
!	    gnexc2(i,ncycle) = gen*gen + gnexc2(i,ncycle)
!	    gninh2(i,ncycle) = gin*gin + gninh2(i,ncycle)

	    vmem(i,ncycle)  = v(i)      + vmem(i,ncycle)
	    vmem2(i,ncycle) = v(i)*v(i) + vmem2(i,ncycle)
	  enddo
!------------------------------------------------------------
	endif
!------------------------------------------------------------
!	if ((t.gt.tfinal-twindow2).and.(mod(iii,10).eq.0)) then
!-----------------------write DIAGNOSTICS for selected neuron
!					    every millisecond
!	  ncount = ncount + 1
!	  do ii=1,100
!	    i = in(ii)
!	    if ( excite(i) ) then
!	      gei = ((1-fnmdac)*ge(i) + fnmdac*gf(i)) * see(i)
!	      gii = ((1-fgaba)*gi(i) + fgaba*gj(i)) * sei(i)
!	    else
!	      gei = ((1-fnmdac)*ge(i) + fnmdac*gf(i)) * sie(i)
!	      gii = ((1-fgaba)*gi(i) + fgaba*gj(i)) * sii(i)
!	    endif
!	    gen = (1-fnmdat)*gx(i) + fnmdat*gy(i)
!	    gin = (1-fgaba)*gn(i) + fgaba*gm(i)
!	    open(7000+ii,status='old',form='unformatted',
!     1		access='direct',recl=iword*8)
!	    write(7000+ii,rec=ncount) v(i),gl(i),gei,gii,gen,gin,
!     1		-alpha1(i),beta1(i)
!	    close(7000+ii)
!	  enddo
!------------------------------------------------------------
!	endif
!------------------------------------------------------------
	if (mod(iii,25*nstep0).eq.0) then
!------------------------------------------------------------
	  if (myid .eq. 0) then
	    print *, 'timestep',iii,' time ',sngl(t),' Nspike ',nsptot, &
		' Spike Rate ',sngl(nsptot/t/nmax),' Inh Rate ', &
		sngl(isptot/t/nmax*4.D0) 
	    print *,' Avgs ',sngl(condavg/nmax), &
		sngl(curravg/nmax),sngl(condlgn/nmax),sngl(condexc/nmax), &
		sngl(condinh/nmax),sngl(geavg/nmax),sngl(giavg/nmax)
	    print *,' Pop Rates ES/EC ',sngl(nessp/t/(0.5*0.75*nmax)), &
		sngl(necsp/t/(0.5*0.75*nmax)),' IS/IC ', &
		sngl(nissp/t/(0.5*0.25*nmax)), &
		sngl(nicsp/t/(0.5*0.25*nmax))
	  endif
!------------------------------------------------------------
	endif
!------------------------------------------------Output to f1
	if (mod(iii,nstep1).eq.0) then
!------------------------------------------------------------
	  ist1 = ist1 + 1
	  open(11,file=f1,status='old',form='unformatted', &
		access='direct',recl=iword2*25*nmax)
	  write(11,rec=1) irate
	  close(11)
	  open(11,file=f1,status='old',form='unformatted', &
		access='direct',recl=iword2)
	  write(11,rec=25*nmax+1) ist1
	  close(11)
	endif
!------------------------------------------------Output to f4
	if ((mod(iii,nstep4).eq.0).and.(t.gt.2.1D0)) then
!------------------------------------------------------------
	  ist4 = ist4 + 1
	  nc = nc + tstep4/period
!	  print *,'Printing to f4, cycle-avg conds, nc = ',nc
	  do j=1,25
	  do i=1,nmax
	    glgn(i,j) = glgn(i,j)/nc
	    gexc(i,j) = gexc(i,j)/nc
	    ginh(i,j) = ginh(i,j)/nc
	    gtot(i,j) = gtot(i,j)/nc
	    cond(i,j) = cond(i,j)/nc
	    vslave(i,j) = vslave(i,j)/nc
	    gnexc(i,j) = gnexc(i,j)/nc
	    gninh(i,j) = gninh(i,j)/nc
	    vmem(i,j)  = vmem(i,j)/nc

	    glgn2(i,j) = glgn2(i,j)/nc
	    gexc2(i,j) = gexc2(i,j)/nc
	    ginh2(i,j) = ginh2(i,j)/nc
	    gtot2(i,j) = gtot2(i,j)/nc
	    cond2(i,j) = cond2(i,j)/nc
	    vslave2(i,j) = vslave2(i,j)/nc
	    gnexc2(i,j) = gnexc2(i,j)/nc
	    gninh2(i,j) = gninh2(i,j)/nc
	    vmem2(i,j) = vmem2(i,j)/nc
	  enddo
	  enddo
	  ist4 = 1
	  open(14,file=f4,status='old',form='unformatted', &
		access='direct',recl=iword*18*nmax*25)
	  write(14,rec=ist4) glgn,gexc,ginh,gtot,cond,vslave,vmem, &
		gnexc,gninh,glgn2,gexc2,ginh2,gtot2,cond2,vslave2, &
		vmem2,gnexc2,gninh2
	  close(14)
          do j=1,25
          do i=1,nmax
            glgn(i,j) = glgn(i,j)*nc
            gexc(i,j) = gexc(i,j)*nc
            ginh(i,j) = ginh(i,j)*nc
            gtot(i,j) = gtot(i,j)*nc
            cond(i,j) = cond(i,j)*nc
            vslave(i,j) = vslave(i,j)*nc
            gnexc(i,j) = gnexc(i,j)*nc
            gninh(i,j) = gninh(i,j)*nc
            vmem(i,j)  = vmem(i,j)*nc

            glgn2(i,j) = glgn2(i,j)*nc
            gexc2(i,j) = gexc2(i,j)*nc
            ginh2(i,j) = ginh2(i,j)*nc
            gtot2(i,j) = gtot2(i,j)*nc
            cond2(i,j) = cond2(i,j)*nc
            vslave2(i,j) = vslave2(i,j)*nc
            gnexc2(i,j) = gnexc2(i,j)*nc
            gninh2(i,j) = gninh2(i,j)*nc
            vmem2(i,j) = vmem2(i,j)*nc
          enddo
          enddo
        endif
!------------------------------------------------------------
      enddo
!------------------------------------End of TimeStepping Loop
!      print *,'Neurons : ',(in(j),j=1,100)
!      do ii=1,100
!        open(7000+ii,status='old',form='unformatted',
!     1          access='direct',recl=iword)
!        write(7000+ii,rec=ncount*8+1) 1.0d0*in(ii)
!        close(7000+ii)
!      enddo

      if (myid .eq. 0) then
      print *,'--------------------------------------------'
      print *,'Integrate-and-Fire Network of ',ni,' x ',nj,' Neurons'
      print *,'--------------------------------------------'
      print *,'         leak  = ',sngl(gleak)
      print *,'       Vthres  = ',sngl(vthres),'    Ve = ',sngl(vexcit)
      print *,'       Vreset  = ',sngl(vreset),'    Vi = ',sngl(vinhib)
      print *,'--------------------------------------------'
      print *,'        spatial coupling : '
      print *,'        den_e  = ',sngl(denexc),' den_i = ',sngl(deninh)
      print *,'        axn_e  = ',sngl(axnexc),' axn_i = ',sngl(axninh)
      print *,'--------------------------------------------'
      print *,' synaptic time constants : '
      print *,'        tau_e  = ',sngl(tauexc)
      print *,'        tau_i  = ',sngl(tauinh),'tau_i-2= ',sngl(tau2)
      print *,'--------------------------------------------'
      print *,'          LGN parameters : '
      print *,'  firing rate  = ',sngl(frtlgn),'     g0 = ',sngl(g0)
      print *,'--------------------------------------------'
      print *,'        noise parameters : '
      print *,' poisson rate  = ',sngl(frtinhE),'    str = ',sngl(ciE0)
      print *,'      (inhib)  = ',sngl(frtinhI),'    str = ',sngl(ciI0)
      print *,'--------------------------------------------'
      print *,'      grating parameters : '
      print *,'   temp. freq  = ',sngl(omega/twopi), &
		'spat. k = ',sngl(gk)
      print *,'        angle  = ',sngl(gtheta),'  phase = ',sngl(gphi)
      print *,'       tstart  = ',sngl(tstart)
      print *,'--------------------------------------------'
      endif

      call MPI_FINALIZE(ierr)
!------------------------------------------------------------
9000  format(1h ,75('-')/1h )
9010  format(1h ,15x,'Welcome to the Integrate-and Fire Code'/1h )
9030  format(1h ,18x,'Size of calculation ',i4,' x ',i4/1h )
9040  format(1h ,10x,'Thres  = ',e10.4,10x,' Reset = ',e10.4/ &
		1h ,10x,'Vexcit = ',e10.4,10x,'Vinhib = ',e10.4/ &
		1h ,10x,'Texcit = ',e10.4,10x,'Tinhib = ',e10.4/ &
		1h ,10x,'  Leak = ',e10.4/ & 
		1h ,10x,'   See = ',e10.4,10x,'   Sei = ',e10.4/ &
		1h ,10x,'   Sie = ',e10.4,10x,'   Sii = ',e10.4/ &
		1h ,10x,'DenExc = ',i6,14x,'AxnExc = ',i6/ &
		1h ,10x,'DenInh = ',i6,14x,'AxnInh = ',i6/1h )
9050  format(1h ,30x,'LGN Parameters'/1h )
9060  format(1h ,10x,'F-rate = ',e10.4,10x,'    g0 = ',e10.4/ &
		1h ,10x,'  tau0 = ',e10.4,10x,'  tau1 = ',e10.4/1h )
9070  format(1h ,29x,'Noise Parameters'/1h )
9080  format(1h ,10x,'FrtExc = ',e10.4,10x,'FrtInh = ',e10.4/ &
		1h ,10x,'  Cexc = ',e10.4,10x,'  Cinh = ',e10.4/1h )
9100  format(1h ,10x,'Ntotal = ',i6,14x,'  Time = ',e10.4/ &
		1h ,10x,'Nstep1 = ',i6,14x,'   dt1 = ',e10.4/ &
		1h ,10x,'Nstep2 = ',i6,14x,'   dt2 = ',e10.4/ &
		1h ,10x,'Nstep3 = ',i6,14x,'   dt3 = ',e10.4/ & 
		1h ,10x,'Nstep4 = ',i6,14x,'   dt4 = ',e10.4/1h )
!9060  format(1h ,'Nit = ',i6/
!     1       1h ,'Nd1 = ',i6/
!     2       1h ,'Nd2 = ',i6/
!     3       1h ,'Nd3 = ',i6/1h )
!------------------------------------------------------------
 9999 stop
      end
!************************************************************
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
!------------------------------------------------------------
      iword2 = iword/2
      if (nspike.gt.0) then
	call disort(tspike,ispike,nspike)

	do i=1,nspike
	  if ( excite(ispike(i)) ) then
	    if ( nlgni(ispike(i)) .gt. 15) then
		nessp = nessp + 1
	    else
		necsp = necsp + 1
	    endif
	  else
	    if ( nlgni(ispike(i)) .gt. 15) then
		nissp = nissp + 1
	    else
		nicsp = nicsp + 1
	    endif
	  endif
	enddo

	if (t.gt.tfinal-twindow) then
	  open(20,file=fsps,status='old',form='unformatted', &
		access='direct',recl=iword2)
	  read(20,rec=1) fnsp
	  nsp = fnsp
	  do i=1,nspike
	    write(20,rec=(nsp+i)*2) sngl(ispike(i)*1.0D0)
	    write(20,rec=(nsp+i)*2+1) sngl(t+tspike(i))
	  enddo
	  fnsp = fnsp + sngl(nspike*1.0D0)
	  write(20,rec=1) fnsp
	  close(20)
	endif

	do i=1,nmax
	  ge(i)  = gep(i)
	  se1(i) = se1p(i)
	  se2(i) = se2p(i)
	  se3(i) = se3p(i)
	  gf(i)  = gfp(i)
	  sf1(i) = sf1p(i)
	  sf2(i) = sf2p(i)
	  sf3(i) = sf3p(i)
	  gi(i)  = gip(i)
	  si1(i) = si1p(i)
	  si2(i) = si2p(i)
	  si3(i) = si3p(i)
	  gj(i)  = gjp(i)
	  sj1(i) = sj1p(i)
	  sj2(i) = sj2p(i)
	  sj3(i) = sj3p(i)
	enddo
	taueratio = tau_e/tnrise
	tauiratio = tau_i/tau2

	tt = tspike(1)
	do j=1,nspike
          call conduc(ge,se1,se2,se3,tau_e,tt,nmax)
          call chain2(gf,sf1,sf2,sf3,tnrise,tndamp,tt,nmax)
          call conduc(gi,si1,si2,si3,tau_i,tt,nmax)
          call conduc(gj,sj1,sj2,sj3,tau2,tt,nmax)
	  ij = ispike(j)
!------------------------------------------------------------
!cc
!cc  calculate delta-function amplitudes given ispike(j)
!cc
	  if (excite(ij)) then
	    do i=1,nmax
	      ii = mod(nmax+i-ij,nmax) + 1
	      cnntvy = icnntvy(ii,indmap(i))*1.0d0
	      if (cnntvy.gt.0.5d0) then
	      if (excite(i)) then
		se3(i) = se3(i) + a_ee(ii) * cnntvy
		sf3(i) = sf3(i) + a_ee(ii)*taueratio * cnntvy
	      else
		se3(i) = se3(i) + a_ie(ii) * cnntvy
		sf3(i) = sf3(i) + a_ie(ii) * taueratio * cnntvy
	      endif
	      endif
	    enddo
	  else
	    do i=1,nmax
	      ii = mod(nmax+i-ij,nmax) + 1
	      cnntvy = icnntvy(ii,indmap(i))*1.0d0
	      if (cnntvy.gt.0.5d0) then
	      if (excite(i)) then
	 	si3(i) = si3(i) + a_ei(ii) * cnntvy
	 	sj3(i) = sj3(i) + a_ei(ii) * tauiratio * cnntvy
	      else
		si3(i) = si3(i) + a_ii(ii) * cnntvy
	 	sj3(i) = sj3(i) + a_ii(ii) * tauiratio * cnntvy
	      endif
	      endif
	    enddo
	  endif
!--------------------------------------Onto next subinterval!
	  tt = tspike(j+1) - tspike(j)
!------------------------------------------------------------
	enddo
!-----------------Update chain between last spike & next time
	call conduc(ge,se1,se2,se3,tau_e,dt-tspike(nspike),nmax)
	call chain2(gf,sf1,sf2,sf3,tnrise,tndamp,dt-tspike(nspike),nmax)
	call conduc(gi,si1,si2,si3,tau_i,dt-tspike(nspike),nmax)
	call conduc(gj,sj1,sj2,sj3,tau2,dt-tspike(nspike),nmax)
      endif

      do i=1,nmax
	gep(i)  =  ge(i)
	se1p(i) = se1(i)
	se2p(i) = se2(i)
	se3p(i) = se3(i)
	gfp(i)  =  gf(i)
	sf1p(i) = sf1(i)
	sf2p(i) = sf2(i)
	sf3p(i) = sf3(i)
	gip(i)  =  gi(i)
	si1p(i) = si1(i)
	si2p(i) = si2(i)
	si3p(i) = si3(i)
	gjp(i)  =  gj(i)
	sj1p(i) = sj1(i)
	sj2p(i) = sj2(i)
	sj3p(i) = sj3(i)
      enddo
!------------------------------------------------------------
      return
      end
!************************************************************
      subroutine e_or_i(excite,exc,iseed)
!------------------------------------------------------------
!  Set up excitatory/inhibitory tag
!  One Quarter of population is inhibitory
!    regular (or random) lattice
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
      logical excite(ni,nj)
      dimension exc(ni,nj)
!------------------------------------------------------------
      do j=1,nj
      do i=1,ni
        excite(i,j) = .true.
	exc(i,j) = 1.D0
!	if ((mod(j,2).eq.1).and.(mod(i,2).eq.0)) excite(i,j) = .false.
	if ((mod(j,2).eq.1).and.(mod(i,2).eq.0)) then
!
! Uncomment line below AND comment line above to make inhibitory 
!    locations random
!	if (ran2(iseed).lt.0.25D0) then
	  excite(i,j) = .false.
	  exc(i,j)    = 0.D0
	endif
      enddo
      enddo
!------------------------------------------------------------
      return
      end
!************************************************************
      subroutine visual(frtlgn0,cond0,frtinhE,ciE0,frtinhI,ciI0,t,dt,iseed)
!------------------------------------------------------------
!     Also generate noise (1-1) given firing rates
!
!     Generates LGN spike times using Poisson process 
!	with time-dependent firing rate a function of
!	visual stimulus
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
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
      common / consat / contr

      omegat = omega*(t-tstart)
      if (t.le.tstart) frate = frtlgn0

      if (frtlgn0.gt.0.d0) then
      do i=1,nlgn
	if (t.gt.pspon(i)+0.0015) then
!-------------Find firing rate as function of visual stimulus
! For CONTRAST REVERSAL, need only amplitude of sinusoid
!------------------------------------------------------------
	if ((t.gt.0.D0).and.(t.lt.tstart)) then
     	  frate = frtlgn0 + (t/tstart) * &
		(ampson(i)*sin(omegat) + ampcon(i)*cos(omegat))
	endif
	if (t.ge.tstart) then
     	  frate = frtlgn0 + ampson(i)*sin(omegat) + ampcon(i)*cos(omegat)
	endif
!------------------------Compute spikes only if non-zero rate
	if (frate.gt.0d0) then
!------------------------------------------------------------
! Contrast Saturation
!------------------------------------------------------------
!	  ifrate = min(int(frate)*10.d0/frtlgn0,199)
!	  r2  = contr(ifrate+1)
!	  r1  = contr(ifrate)
!	  satur  = r1 + (r2-r1) * (frate*10.d0/frtlgn0-ifrate)
!c	  if (i.eq.1) print *,sngl(frate),sngl(satur),ifrate
!	  frate  = satur*frtlgn0/10.d0
!------------------------------------------------------------
	  rannum = ran2(iseed)
	  if (rannum.lt.dt*frate) then
	    pspon(i) = t + rannum/frate

	    glo(i)  = glop(i) 
	    slo1(i) = slo1p(i)
	    slo2(i) = slo2p(i)
	    slo3(i) = slo3p(i)
	    glon(i)  = glonp(i) 
	    slon3(i) = slon3p(i)

	    dtt = rannum/frate
	    te  = dtt/tau_e
	    te2 = te*te/2.
	    te3 = te*te2/3.
	    ete = exp(-te)
	    glo(i)  = (glo(i)  + slo3(i)) * ete
	    slo3(i) = (slo3(i)          ) * ete + cond0

	    tr  = dtt/tnrise
	    etr = exp(-tr)
	    td  = dtt/tndamp
	    etd = exp(-td)
	    const = tnrise/(tndamp - tnrise) * (etd - etr)
	    glon(i)  =  glon(i) * etd + const * slon3(i)
	    slon3(i) = slon3(i) * etr + cond0*tau_e/tnrise

	    te  = (dt-dtt)/tau_e
	    te2 = te*te/2.
	    te3 = te*te2/3.
	    ete = exp(-te)
	    glo(i)  = (glo(i)  + slo3(i)*te)*ete
	    slo3(i) = (slo3(i)             )*ete 

	    tr  = (dt-dtt)/tnrise
	    etr = exp(-tr)
	    td  = (dt-dtt)/tndamp
	    etd = exp(-td)
	    const = tnrise/(tndamp - tnrise) * (etd - etr)
	    glon(i)  =  glon(i) * etd + const * slon3(i)
	    slon3(i) = slon3(i) * etr
	  endif
	endif
	endif
	glop(i)  = glo(i) 
	slo1p(i) = slo1(i)
	slo2p(i) = slo2(i)
	slo3p(i) = slo3(i)
	glonp(i) = glon(i) 
	slon3p(i) = slon3(i)
      enddo
!----------------------------------Now off-centered LGN cells
      do i=1,nlgn
	if (t.gt.pspoff(i)+0.0015) then
	if ((t.gt.0.D0).and.(t.lt.tstart)) then
     	  frate = frtlgn0 + (t/tstart) * & 
		(ampsof(i)*sin(omegat) + ampcof(i)*cos(omegat))
	endif
	if (t.ge.tstart) then
     	  frate = frtlgn0 + ampsof(i)*sin(omegat) + ampcof(i)*cos(omegat)
	endif
!------------------------------------------------------------
	if (frate.gt.0.d0) then
!------------------------------------------------------------
! Contrast Saturation
!------------------------------------------------------------
!	  ifrate = min(int(frate)*10.d0/frtlgn0,199)
!	  r2  = contr(ifrate+1)
!	  r1  = contr(ifrate)
!	  satur  = r1 + (r2-r1) * (frate*10.d0/frtlgn0-ifrate)
!c	  if (i.eq.1) print *,sngl(frate),sngl(satur),ifrate
!	  frate  = satur*frtlgn0/10.d0
!------------------------------------------------------------
	  rannum = ran2(iseed)
	  if (rannum.lt.dt*frate) then
	    pspoff(i) = t + rannum/frate
	    glf(i)  = glfp(i) 
	    slf1(i) = slf1p(i)
	    slf2(i) = slf2p(i)
	    slf3(i) = slf3p(i)
	    glfn(i)  = glfnp(i) 
	    slfn3(i) = slfn3p(i)

	    dtt = rannum/frate
	    te  = dtt/tau_e
	    te2 = te*te/2.
	    te3 = te*te2/3.
	    ete = exp(-te)
	    glf(i)  = (glf(i)  + slf3(i)*te) * ete
	    slf3(i) = (slf3(i)             ) * ete + cond0

	    tr  = dtt/tnrise
	    etr = exp(-tr)
	    td  = dtt/tndamp
	    etd = exp(-td)
	    const = tnrise/(tndamp - tnrise) * (etd - etr)
	    glfn(i)  =  glfn(i) * etd + const * slfn3(i)
	    slfn3(i) = slfn3(i) * etr + cond0*tau_e/tnrise

	    te  = (dt-dtt)/tau_e
	    te2 = te*te/2.
	    te3 = te*te2/3.
	    ete = exp(-te)
	    glf(i)  = (glf(i)  + slf3(i)*te)*ete
	    slf3(i) = (slf3(i)             )*ete 

	    tr  = (dt-dtt)/tnrise
	    etr = exp(-tr)
	    td  = (dt-dtt)/tndamp
	    etd = exp(-td)
	    const = tnrise/(tndamp - tnrise) * (etd - etr)
	    glfn(i)  =  glfn(i) * etd + const * slfn3(i)
	    slfn3(i) = slfn3(i) * etr
	  endif
	endif
	endif
	glfp(i)  = glf(i) 
	slf1p(i) = slf1(i)
	slf2p(i) = slf2(i)
	slf3p(i) = slf3(i)
	glfnp(i)  = glfn(i) 
	slfn3p(i) = slfn3(i)
      enddo
      endif
!----------------------------------Now inhibitory noise units
      frate = frtinhE
      if (frate.gt.0.0d0) then
      do i=1,nmax
	if ( excite(i) ) then
	  frate = frtinhE
	  ci0 = ciE0
	else
	  frate = frtinhI
	  ci0 = ciI0
	endif

	tlocal = t

	gn(i)   = gnp(i)
	sn1(i)  = sn1p(i)
	sn2(i)  = sn2p(i)
	sn3(i)  = sn3p(i)
	gm(i)   = gmp(i)
	sm1(i)  = sm1p(i)
	sm2(i)  = sm2p(i)
	sm3(i)  = sm3p(i)


 102    continue

        if (t+dt.gt.pspinh(i)) then

          dtt = pspinh(i) - tlocal

	  ti  = dtt/tau_i
	  ti2 = ti*ti/2.
	  ti3 = ti*ti2/3.
	  eti = exp(-ti)
	  tj  = dtt/tau2
	  tj2 = tj*tj/2.
	  tj3 = tj*tj2/3.
	  etj = exp(-tj)

	  gn(i)  = (gn(i)  + sn3(i)*ti)*eti
	  sn3(i) = (sn3(i)            )*eti + ci0/tau_i

	  gm(i)  = (gm(i)  + sm3(i)*tj)*etj
	  sm3(i) = (sm3(i)            )*etj + ci0/tau2

          tlocal = pspinh(i)
          rannum = ran2(iseed)
          tnext  = -dlog(rannum)/frate
          pspinh(i) = pspinh(i) + tnext
          goto 102
        endif

        dtt = t + dt - tlocal

	ti  = dtt/tau_i
	ti2 = ti*ti/2.
	ti3 = ti*ti2/3.
	eti = exp(-ti)
	tj  = dtt/tau2
	tj2 = tj*tj/2.
	tj3 = tj*tj2/3.
	etj = exp(-tj)

	gn(i)  = (gn(i)  + sn3(i)*ti)*eti
	sn3(i) = (sn3(i)            )*eti
	gm(i)  = (gm(i)  + sm3(i)*tj)*etj
	sm3(i) = (sm3(i)            )*etj

	gnp(i)  = gn(i)
	sn1p(i) = sn1(i)
	sn2p(i) = sn2(i)
	sn3p(i) = sn3(i)
	gmp(i)  = gm(i)
	sm1p(i) = sm1(i)
	sm2p(i) = sm2(i)
	sm3p(i) = sm3(i)
      enddo
      endif
!------------------------------------------------------------
 999  return
      end
!************************************************************
      subroutine chain(ge,se1,se2,se3,tau,deltat,nn)
!------------------------------------------------------------
!  Oct 2000: Updating chain w/o intracortical spikes
!    PSP handled in rk2/rk4 routine
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      dimension ge(nn),se1(nn),se2(nn),se3(nn)
!------------------------------------------------------------
      te  = deltat/tau
      te2 = te*te/2.
      te3 = te*te2/3.
      ete = exp(-te)

      do i=1,nn
!        ge(i)  = (ge(i) + se1(i)*te + se2(i)*te2
!     1         +  se3(i)*te3)*ete
!        se1(i) = (se1(i) + se2(i)*te + se3(i)*te2)*ete
!        se2(i) = (se2(i) + se3(i)*te)*ete
        ge(i)  = (ge(i) + se3(i)*te) * ete
        se3(i) = (se3(i)           ) * ete
      enddo
!------------------------------------------------------------
      return
      end
!************************************************************
      subroutine chain2(ge,se1,se2,se3,trise,tdamp,deltat,nn)
!------------------------------------------------------------
!  Oct 2000: Updating difference of expon'tial w/o intracortical spikes
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      dimension ge(nn),se1(nn),se2(nn),se3(nn)
!------------------------------------------------------------
      tr  = deltat/trise
      etr = exp(-tr)

      td  = deltat/tdamp
      etd = exp(-td)

      const = trise/(tdamp - trise) * (etd - etr)

      do i=1,nn
	ge(i)  =  ge(i) * etd + const * se3(i)
        se3(i) = se3(i) * etr
      enddo
!------------------------------------------------------------
      return
      end
!************************************************************
      subroutine gaussk(prefct,bb,aa,nx,ny,al2,dx2,dy2,fglobal)
!------------------------------------------------------------
!  Oct 2000: modified for chain (do NOT fourier transform)
!
!    bb : gaussian kernel on exit
!    aa : work array
!    bb = prefct/al2 * exp(-dd/al2), dd = (i-1)*(i-1)*dx2 + (j-1)*(j-1)*dy2
!         Gaussian centered at (1,1)
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      dimension aa(nx*ny),bb(nx,ny)
!------------------------------------------------------------
      ni = nx/2
      nj = ny/2
      const = sqrt(dx2*dy2)/nx/ny/4.0d0/atan(1.0)

      sum = 0.0d0
      do j=1,ny
      do i=1,nx
	dd = (i-ni-1)*(i-ni-1)*dx2 + (j-nj-1)*(j-nj-1)*dy2
        bb(i,j) = const/al2*exp(-dd/al2)
      enddo
      enddo
      do j=1,ny
      do i=1,nx
        jj = (j-1)*nx + i
        ii = mod(nx*ny+jj-(nx*ny/2+nx/2+1),nx*ny) + 1
        aa(ii) = bb(i,j)
        sum = sum + aa(ii)
      enddo
      enddo

      ccc = sum*nx*ny

      do j=1,ny
      do i=1,nx
        jj = (j-1)*nx + i
        bb(i,j) = prefct*aa(jj)/sum
	aa(jj) = prefct/nx/ny
	bb(i,j) = (1-fglobal)*bb(i,j) + fglobal*aa(jj)
      enddo
      enddo
!------------------------------------------------------------
      return
      end
!************************************************************
      subroutine lgnrf(frtlgn)
!------------------------------------------------------------
!  Map each LGN cell to its receptive field :
!	location & spatiotemporal filter parameters
!
!  Determine each cell's firing rate given visual stimulus :
!
!  	f(t)   = f0 + \int_0^t ds \int dy G(t - s) A(x - y) I(y,s)
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
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
      parameter ( nlx = 48 , nly = 64 , nlgn = nlx*nly , maxlgn = 60 )
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
      dimension xlgn(nlx*nly),ylgn(nlx*nly)
      dimension ampcon(nlx*nly),ampson(nlx*nly)
      dimension ampcof(nlx*nly),ampsof(nlx*nly)
      common / lgnpos / xlgn,ylgn
      common / lgnrfs / ampson,ampsof,ampcon,ampcof
      common / conrev / omega,gkx,gky,gphi,tstart,contrast
!------------------------------------------------------------
!  Hardwire each LGN cell
!------------------------------------------------------------
      tau0 = 0.003D0
      tau1 = 0.005D0
      ot0  = omega*tau0
      ot1  = omega*tau1
      ampc = 240.D0*((3 - 10*ot0*ot0 + 3*(ot0**4))*ot0 / &
	(1+ot0*ot0)**6 - (3 - 10*ot1*ot1 + 3*(ot1**4))*ot1 / &
	(1+ot1*ot1)**6)
      amps = 120.D0*((-1 + 15*ot0*ot0 - 15*(ot0**4) + ot0**6) &
	* ot0 / (1+ot0*ot0)**6 & 
	- (-1 + 15*ot1*ot1 - 15*(ot1**4) + ot1**6) &
	* ot1 / (1+ot1*ot1)**6)
      tmpphi = atan2(amps,ampc)
      gk2    = gkx*gkx + gky*gky
      gk2opt = 10.d0*10.d0
      siga2  = 1.0d0/40./40.
      sigb2  = 1.5d0*1.5d0 * siga2
      dogopt = exp(-gk2opt*siga2/4.d0) - 0.84d0*exp(-gk2opt*sigb2/4.d0)
      doggk2 = exp(-gk2   *siga2/4.d0) - 0.84d0*exp(-gk2   *sigb2/4.d0)
!------------------------------------------------------------
      ampccc = ampc/cos(tmpphi) * frtlgn/33.56D0*14.d0
      ampccc = ampccc * doggk2/dogopt * contrast
      print *,'ampccc = ',ampccc
!------------------------------------------------------------
      do i=1,nlgn
        ampson(i)  = -ampccc*cos(gphi-gkx*xlgn(i)-gky*ylgn(i))
	ampsof(i)  = -ampson(i)

        ampcon(i)  =  ampccc*sin(gphi-gkx*xlgn(i)-gky*ylgn(i))
	ampcof(i)  = -ampcon(i)
      enddo
!------------------------------------------------------------
      return
      end
!************************************************************
      subroutine lgnsatur
!------------------------------------------------------------
! Melinda's LGN contrast saturation function
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      dimension contr(200)
!------------------------------------------------------------
      common / consat / contr
!------------------------------------------------------------
      c2 = 0.09827
      c3 = -0.001589

      satmax = 60
      join=30
      d1 = 40
      d4 = satmax-d1*pi/2
      zz = tan((join - d4)/d1) + pi
      d2 = (zz*zz+1)/d1
      d3 = zz - join * d2

      do i=1,40
	rr = i
	contr(i) = c3*(rr**3) + c2*(rr**2)
      enddo

      do i=41,200
	rr = i
	contr(i) = d1 * atan(d2*rr + d3) + d4
      enddo

!      do i=200,21,-1
!	contr(i) = contr(i-1)
!      enddo
!      do i=1,20
!	contr(i) = 0.0
!      enddo
!------------------------------------------------------------
      return
      end
!************************************************************
      subroutine genmap(icnntvy,indmap,pconnect,iseed,myid)
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
      parameter ( nlx = 48 , nly = 64 , nlgn = nlx*nly)
      parameter ( nn = nmax , nmap = 32 )
!------------------------------------------------------------
!  Generate Postsynaptic Target Map
!------------------------------------------------------------
      dimension icnntvy(nmax,nmap),ind(nmax),indmap(nmax)
      data iword / 8 /
!------------------------------------------------------------
      nx1 = ni/2.25d0 + 1
      ny1 = nj/2.25d0 + 1
      nx2 = ni - ni/2.25d0 - 1
      ny2 = nj - nj/2.25d0 - 1
!      nx1 = ni/4 + 1
!      ny1 = nj/4 + 1
!      nx2 = ni - ni/4 - 1
!      ny2 = nj - nj/4 - 1

      do i=1,nmap

	do j=1,nmax
	  icnntvy(j,i) = 0
	enddo

	nconn = pconnect*nmax
!	print *,i,' genmap: Nconn = ',nconn

	do j=1,nconn
 17	  ix = ran2(iseed) * ni + 1
	  iy = ran2(iseed) * nj + 1
	  if (ix.gt.nx1.and.ix.lt.nx2) goto 17
	  if (iy.gt.ny1.and.iy.lt.ny2) goto 17
	  ind(j) = (iy-1)*ni + ix
	  do jj=1,j-1
	    if (ind(j) .eq. ind(jj)) goto 17
	  enddo
	enddo

	do j=1,nconn
	  icnntvy(ind(j),i) = 1
	enddo
      enddo

      do j=1,nmax
 19	ind(j) = ran2(iseed) * nmax
	indmap(j) = mod(ind(j),nmap) + 1
      enddo
!------------------------------------------------------------
      if (myid .eq. 0) then
      open(20,file='map.dat',status='new',form='unformatted', &
		access='direct',recl=nmax*iword/2)
!      print *,'Writing Map Index'
!      write(20,rec=1) indmap
!      print *,'Writing Connectivity Maps'
      do i=1,nmap
	write(20,rec=i+1) (icnntvy(j,i),j=1,nmax)
      enddo
      close(20)
      endif
!------------------------------------------------------------
      return
      end
!************************************************************
