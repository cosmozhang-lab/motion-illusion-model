c***********************************************************
c
c  6/09 Sparse coupling
c
c  5/24 2 Pop Model, LGN density  16 x 16 in 1 mm^2
c    global inhibition, synaptic failure, variable pdrive
c
c  Integrate-and-Fire Model Cortex
c
c  Background Constant Rate
c
c************************************************************
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
c      parameter ( nlx = 24 , nly = 32 , nlgn = nlx*nly , maxlgn = 60 )
      parameter ( nlx = 48 , nly = 64 , nlgn = nlx*nly , maxlgn = 60 )
      parameter ( nmap = 32 )
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
      dimension in(100),nin(100)
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
      dimension icnntvy(nmax,nmap),indmap(nmax)
      logical excite(nmax)
      character*80 f0,f1,f2,f3,f4,finput,fn(8),flgn
      real*4 fnsp
c------------------------------------------------------------
c
c  For each V1 cell ---
c   1) see, sei, sie, sii: intracortical coupling strengths
c   2) gl: summed LGN conductance (summing over all on & off LGN cells
c   3) nonlgn, noflgn: number of On and Off LGN cells sending afferents
c   4) ionlgn, ioflgn: index of On and Off LGN cells sending afferents
c   5) icnntvy: nmap indices of postsynaptic targets
c   6) indmap: (random) choice connectivity map for each V1 cell
c
c  For each LGN cell
c   xlgn, ylgn: coordinates of the center of its "receptive field"
c
c------------------------------------------------------------
      common / smatrx / see,sei,sie,sii
      common / lgncnd / gl
      common / lgnmap / nonlgn,noflgn,ionlgn,ioflgn
      common / lgnpos / xlgn,ylgn
      common / sparse / icnntvy,indmap
c------------------------------------------------------------
c
c  glo, glon: On-centered LGN cells, AMPA & NMDA channels
c  glf, glfn: Off-centered LGN cells, AMPA & NMDA channels
c
c------------------------------------------------------------
      common / chaino / glo,slo1,slo2,slo3,glop,slo1p,slo2p,slo3p
      common / chaino2/ glon,slon1,slon2,slon3,
     1		glonp,slon1p,slon2p,slon3p
      common / chainf / glf,slf1,slf2,slf3,glfp,slf1p,slf2p,slf3p
      common / chainf2/ glfn,slfn1,slfn2,slfn3,
     1		glfnp,slfn1p,slfn2p,slfn3p
c------------------------------------------------------------
c
c  ge, gf: intracortical excitation, AMPA & NMDA, resp.
c  gi, gj: intracortical inhibition, GABA_A & B, resp.
c
c------------------------------------------------------------
      common / chaine / ge,se1,se2,se3,gep,se1p,se2p,se3p
      common / chaine2/ gf,sf1,sf2,sf3,gfp,sf1p,sf2p,sf3p
      common / chaini / gi,si1,si2,si3,gip,si1p,si2p,si3p
      common / chainj / gj,sj1,sj2,sj3,gjp,sj1p,sj2p,sj3p
c------------------------------------------------------------
c
c  gx, gy: "background" excitation, AMPA & NMDA
c  gn, gm: "background" inhibition, GABA_A & B
c
c------------------------------------------------------------
      common / chainx / gx,sx1,sx2,sx3,gxp,sx1p,sx2p,sx3p
      common / chainy / gy,sy1,sy2,sy3,gyp,sy1p,sy2p,sy3p
      common / chainn / gn,sn1,sn2,sn3,gnp,sn1p,sn2p,sn3p
      common / chainm / gm,sm1,sm2,sm3,gmp,sm1p,sm2p,sm3p
c------------------------------------------------------------
c  Various reversal potentials and time constants
c------------------------------------------------------------
      common / vconst / vthres,vreset,vexcit,vinhib,gleak
      common / tconst / tau_e,tau_i,tau0,tau1,tau2,tnrise,tndamp
c------------------------------------------------------------
c  fnmdat(c) Fraction of NMDA for thalamocortical (& intracortical)
c  fgaba     Fraction of GABA_B
c------------------------------------------------------------
      common /  NMDA  / fnmdat,fnmdac,fgaba
c------------------------------------------------------------
c  a_xy: spatial kernels
c  excite:  = 1 for excitatory neurons, = 0 for inhibitory
c  nlgni:  No. of LGN afferents
c------------------------------------------------------------
      common / kernel / a_ee,a_ei,a_ie,a_ii
      common / neuron / excite,nlgni
c------------------------------------------------------------
c  Stimulus parameters:
c    omega: temporal frequency
c    (gkx,gky): grating direction vector
c    gphi: grating spatial phase
c
c    contr: array for LGN contrast saturation
c------------------------------------------------------------
      common / conrev / omega,gkx,gky,gphi,tstart,contrast
      common / consat / contr
      common /   rhs  / alpha1,beta1
      common /  avgs  / condavg,curravg,condlgn,condexc,condinh,
     1		geavg,giavg
      common /   isi  / pspike,nsptot,isptot,nessp,necsp,nissp,nicsp
      common / spikes / irate,period
      common / lgnsps / pspon,pspoff
      common /  psps  / pspexc,pspinh
      common / ttotal / tfinal,twindow
      common / synapt / pconn
      data iword / 8 /
cSGI      data iword / 2 /
      external chain
c------------------------------------------------------------
c
c  Some (not all) INPUT & OUTPUT file declarations
c
c------------------------------------------------------------
      iword2 = iword/2
      iseed0 = 22594
      finput = 'INPUT'
      f0     = 'i-and-f.list'
      f1     = 'i-and-f.dat1'
      f2     = 'i-and-f.dat2'
      f3     = 'i-and-f.dat3'
      f4     = 'i-and-f.dat4'
      flgn   = 'lgnmap.out'
c------------------------------------------------------------
c
c Read INPUT file for run parameters
c
c------------------------------------------------------------
      open(3,file=finput,access='sequential',form='formatted')
      read(3,*)
      read(3,*)
      read(3,*)
      read(3,*)
      read(3,*)
      read(3,*) dt,tfinal,tstep1,tstep2,tstep3,tstep4,iseed
      read(3,*)
      read(3,*)
      read(3,*) vthres,vreset,vexcit,vinhib,gleak,fglobal,fglobali
      read(3,*)
      read(3,*)
      read(3,*) tau_e,tau_i,tnrise,tndamp,tau2,pconn
      read(3,*)
      read(3,*)
      read(3,*) denexc,axnexc,deninh,axninh,fnmdat,fnmdac,fgaba
      read(3,*)
      read(3,*)
      read(3,*) frtlgn,g0,tau0,tau1,frtexc,ce0,frtinh,ci0
      read(3,*)
      read(3,*)
      read(3,*) omega,gk,gtheta,gphi,tstart,twindow,contrast
      read(3,*)
      read(3,*)
      read(3,*) seemax,seimax,siemax,siimax,ceemax,ceimax,ciemax,ciimax
      close(3)
      iseed1 = iseed0 + iseed	
c------------------------------------------------------------
      if (tstart .lt. 0.1D0) tstart = 0.1D0
c------------------------------------------------------------
c     Read in LGN location and LGN-V1 map from file flgn
c------------------------------------------------------------
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
c	rannum = ran2(iseed)
c        if (rannum .gt. pdrive) then
c          nonlgn(i) = 0
c          noflgn(i) = 0
c        endif
        nlgni(i) = nonlgn(i) + noflgn(i)
      enddo
      close(14)
c------------------------------------------------------------
c     Initialize output files
c------------------------------------------------------------
      open(20,file='spikes.dat',status='new',form='unformatted',
     1       access='direct',recl=iword2)
      fnsp = sngl(0.0d0)
      write(20,rec=1) fnsp
      close(20)

      ist1 = 0
      open(11,file=f1,status='new',form='unformatted',
     1       access='direct',recl=iword2*25*nmax)
      close(11)

      ist2 = 0
      open(12,file=f2,status='new',form='unformatted',
     1       access='direct',recl=iword*2*1001)
      close(12)

      ist3 = 0
      open(13,file=f3,status='new',form='unformatted',
     1       access='direct',recl=iword*nmax)
      close(13)

      ist4 = 0
      nc   = 0
      open(14,file=f4,status='new',form='unformatted',
     1       access='direct',recl=iword*nmax*6*25)
      close(14)

      ncount = 0
      do ii=1,100
	nin(ii) = 0
	open(7000+ii,status='new',form='unformatted',
     1       access='direct',recl=iword*8)
	close(7000+ii)
      enddo
c------------------------------------------------------------
c     Initialize spatial constants and spatial kernels
c------------------------------------------------------------
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
c
c  cond0 is normalized to maximum g0 for n_LGN = 15
c
      cond0  =  g0/frtlgn/tau_e/30.D0
      do i=1,nmax
c
c linear S between nlgn = 0 & nlgn = 30
c
	fee = 1.0D0 + (ceemax-seemax)*(1.0D0-nlgni(i)/30.D0)/seemax
	fei = 1.0D0 + (ceimax-seimax)*(1.0D0-nlgni(i)/30.D0)/seimax
	fie = 1.0D0 + (ciemax-siemax)*(1.0D0-nlgni(i)/30.D0)/siemax
	fii = 1.0D0 + (ciimax-siimax)*(1.0D0-nlgni(i)/30.D0)/siimax

	see(i) = seemax * fee
	sei(i) = seimax * fei
	sie(i) = siemax * fie
	sii(i) = siimax * fii
	see(i) = see(i) / tau_e * 4.0d0/3. / pconn
	sei(i) = sei(i) / tau_i * 4.0d0    / pconn
	sie(i) = sie(i) / tau_e * 4.0d0/3. / pconn
	sii(i) = sii(i) / tau_i * 4.0d0    / pconn
      enddo
c------------------------------------------------------------
c
c  Now pick 100 neurons at random to keep conductances/potentials
c
c------------------------------------------------------------
      do j=1,100
 17     in(j) = ran2(iseed0) * nmax
        do jj=1,j-1
          if (in(j) .eq. in(jj)) goto 17
        enddo
      enddo
c------------------------------------------------------------
      print *,'Neurons : '
      print *,(in(j),j=1,100)
      call genmap(icnntvy,indmap,pconn,iseed0)
c------------------------------------------------------------
      call gaussk(1.D0,a_ee,aa,ni,nj,alee2,dx2,dy2,fglobali)
      call gaussk(1.D0,a_ei,aa,ni,nj,alei2,dx2,dy2,fglobal)
      call gaussk(1.D0,a_ie,aa,ni,nj,alie2,dx2,dy2,fglobali)
      call gaussk(1.D0,a_ii,aa,ni,nj,alii2,dx2,dy2,fglobali)

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
      print *,'          See   = ',sngl(seemax),
     1		'    Sie = ',sngl(siemax)
      print *,'          Sei   = ',sngl(seimax),
     1		'    Sii = ',sngl(siimax)
      print *,'--------------------------------------------'
      print *,'        noise parameters : '
      print *,' poisson rate  = ',sngl(frtexc),'    str = ',sngl(ce0)
      print *,'      (inhib)  = ',sngl(frtinh),'    str = ',sngl(ci0)
      print *,'--------------------------------------------'
      print *,'      grating parameters : '
      print *,' spatial freq  = ',sngl(omega), '      k = ',sngl(gk)
      print *,'        angle  = ',sngl(gtheta),'  phase = ',sngl(gphi)
      print *,'       tstart  = ',sngl(tstart),
     1		'twindow = ',sngl(twindow)
      print *,'--------------------------------------------'
      print *,'output and time-stepping : '
      print *,'       tfinal  = ',sngl(tfinal),'     dt = ',sngl(dt)
      print *,'       tstep1  = ',sngl(tstep1),' tstep2 = ',sngl(tstep2)
      print *,'       tstep3  = ',sngl(tstep3),' tstep4 = ',sngl(tstep4)
      print *,'--------------------------------------------'

      ntotal = tfinal/dt
      nstep1 = tstep1/dt
      nstep2 = tstep2/dt
      nstep3 = tstep3/dt
      nstep4 = tstep4/dt
      period = 1.0/omega
      nstep0 = period/dt/25.0
c------------------------------------------------------------
c  Grating parameters: Drift frequency, spatial frequency
c------------------------------------------------------------
      omega  = twopi*omega
      gkx    = gk*cos(gtheta)
      gky    = gk*sin(gtheta)
c------------------------------------------------------------
c
c  Input omega in Hertz
c
c----------------------Determine DRIFTING GRATING time course
c				   E/I location can be random
      call lgnrf(frtlgn)
      call lgnsatur
      call e_or_i(excite,exc,iseed0)
      open(13,file=f3,status='old',form='unformatted',
     1       access='direct',recl=iword*nmax)
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
      print *,'  No. of E/I Cells in Left column : ',int(c1),
     1		nmax/2-int(c1)
      print *,'  No. of E/I Cells in Right column : ',int(count-c1),
     1		nmax/2-int(count-c1)
c---------------------------------------Transient noise & lgn 
      do i=1,nlgn
	pspon(i)  = -10000.0
	pspoff(i) = -10000.0
      enddo
c---------------------------------Total transient 0.1 seconds
      t = -0.1D0
      ntrans = abs(t)/dt
      do i=1,nmax
	if (frtexc.gt.0.0.and.ce0.gt.0.0D0) then
	  pspexc(i) =  t - dlog(ran2(iseed))/frtexc
	endif
	if (frtinh.gt.0.0.and.ci0.gt.0.0D0) then
	  pspinh(i) =  t - dlog(ran2(iseed))/frtinh
	endif
      enddo

      do i=1,ntrans
	call visual(frtlgn,cond0,frtexc,ce0,frtinh,ci0,t,dt,iseed)
	call chain(glo,slo1,slo2,slo3,tau_e,dt,nlgn)
	call chain2(glon,slon1,slon2,slon3,tnrise,tndamp,dt,nlgn)
	call chain(glf,slf1,slf2,slf3,tau_e,dt,nlgn)
	call chain2(glfn,slfn1,slfn2,slfn3,tnrise,tndamp,dt,nlgn)
	call chain(gx,sx1,sx2,sx3,tau_e,dt,nmax)
	call chain2(gy,sy1,sy2,sy3,tnrise,tndamp,dt,nmax)
	call chain(gn,sn1,sn2,sn3,tau_i,dt,nmax)
	call chain(gm,sm1,sm2,sm3,tau2,dt,nmax)
	t = t + dt
      enddo
      print *,'After generating initial transients',t
      t      = 0.0d0

      nsptot = 0
      isptot = 0

      open(10,file=f0,status='new',form='formatted')
      write(10,9000)
      write(10,9000)
      write(10,9010)
      write(10,9000)
      write(10,9000)
      write(10,9030) ni,nj
      write(10,9040) vthres,vreset,vexcit,vinhib,tau_e,tau_i,
     1	gleak,seemax,seimax,siemax,siimax,
     2  int(denexc),int(axnexc),int(deninh),int(axninh)
      write(10,9000)
      write(10,9050) 
      write(10,9060) frtlgn,g0,tau0,tau1
      write(10,9000)
      write(10,9070) 
      write(10,9080) frtexc,frtinh,ce0,ci0
      write(10,9000)
      write(10,9100) ntotal,tfinal,nstep1,tstep1,nstep2,tstep2,
     1	nstep3,tstep3,nstep4,tstep4
      write(10,9000)

      close(10)

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
	  gl(i) = gl(i) + (1-fnmdat)*glo(ionlgn(i,j))
     1		+ fnmdat*glon(ionlgn(i,j))
	enddo
	do j=1,noflgn(i)
	  gl(i) = gl(i) + (1-fnmdat)*glf(ioflgn(i,j))
     1		+ fnmdat*glfn(ioflgn(i,j))
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
	beta1(i)  = (gl(i) + gexcit + genoise)*vexcit 
     1		+ (gnoise + ginhib)*vinhib
	condavg   =  condavg - alpha1(i)
	curravg   =  curravg + beta1(i)
	condlgn   =  condlgn + gl(i)
	condinh   =  condinh + ginoise
	condexc   =  condexc + genoise
	geavg     =  geavg   + gexcit
	giavg     =  giavg   + ginhib
      enddo
      print *,condavg/nmax,curravg/nmax,condlgn/nmax,condexc/nmax,
     1		condinh/nmax,geavg/nmax,giavg/nmax

      print *,'Gm/n avg = ',gmavg/nmax,gnavg/nmax
      print *,'Gx/y avg = ',gxavg/nmax,gyavg/nmax

      print *
      print *,'      tauNMDA = ',sngl(tnrise),sngl(tndamp)
      print *,'  fNMDA tc/cc = ',sngl(fnmdat),sngl(fnmdac)
c-------------Start of Main Time Integration Loop for each dt
      do iii=1,ntotal
c------------------------------------------------------------
ccc	call rk2(chain,v,vnew,nspike,ispike,tspike,t,dt)
	call rk4(chain,v,vnew,nspike,ispike,tspike,t,dt)
c-------------------------Generate noise and LGN input spikes 
	call visual(frtlgn,cond0,frtexc,ce0,frtinh,ci0,t,dt,iseed)
	call update(chain,nspike,ispike,tspike,dt,t,iseed)
c----------------------------------------finally advance TIME
	t = t + dt
	do i=1,nmax
	  v(i) = vnew(i)
	enddo
c------------------------------------Cycle Avg'd Conductances
c					       output file f4
	if ((mod(iii,nstep0).eq.0).and.(t.gt.2.0d0)) then
c------------------------------------------------------------
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
c	    gen = (1-fnmda)*gx(i) + fnmda*gy(i)
c	    gin = 0.5D0 * (gm(i) + gn(i))

	    vsi = -beta1(i)/alpha1(i)
	    glgn(i,ncycle) = gl(i) + glgn(i,ncycle)
	    gexc(i,ncycle) = gei + gexc(i,ncycle)
	    ginh(i,ncycle) = gii + ginh(i,ncycle)
	    gtot(i,ncycle) = -alpha1(i) + gtot(i,ncycle)
	    cond(i,ncycle) = beta1(i) + cond(i,ncycle)
	    vslave(i,ncycle) = vsi + vslave(i,ncycle)
c	    gnexc(i,ncycle) = gen + gnexc(i,ncycle)
c	    gninh(i,ncycle) = gin + gninh(i,ncycle)

	    glgn2(i,ncycle) = gl(i)*gl(i) + glgn2(i,ncycle)
	    gexc2(i,ncycle) = gei*gei + gexc2(i,ncycle)
	    ginh2(i,ncycle) = gii*gii + ginh2(i,ncycle)
	    gtot2(i,ncycle) = alpha1(i)*alpha1(i) + gtot2(i,ncycle)
	    cond2(i,ncycle) = beta1(i)*beta1(i) + cond2(i,ncycle)
	    vslave2(i,ncycle) = vsi*vsi + vslave2(i,ncycle)
c	    gnexc2(i,ncycle) = gen*gen + gnexc2(i,ncycle)
c	    gninh2(i,ncycle) = gin*gin + gninh2(i,ncycle)

	    vmem(i,ncycle)  = v(i)      + vmem(i,ncycle)
	    vmem2(i,ncycle) = v(i)*v(i) + vmem2(i,ncycle)
	  enddo
c------------------------------------------------------------
	endif
c------------------------------------------------------------
	if ((t.gt.tfinal-twindow).and.(mod(iii,10).eq.0)) then
c-----------------------write DIAGNOSTICS for selected neuron
c					    every millisecond
	  ncount = ncount + 1
	  do ii=1,100
	    i = in(ii)
	    if ( excite(i) ) then
	      gei = ((1-fnmdac)*ge(i) + fnmdac*gf(i)) * see(i)
	      gii = ((1-fgaba)*gi(i) + fgaba*gj(i)) * sei(i)
	    else
	      gei = ((1-fnmdac)*ge(i) + fnmdac*gf(i)) * sie(i)
	      gii = ((1-fgaba)*gi(i) + fgaba*gj(i)) * sii(i)
	    endif
	    gen = (1-fnmdat)*gx(i) + fnmdat*gy(i)
	    gin = (1-fgaba)*gn(i) + fgaba*gm(i)
	    open(7000+ii,status='old',form='unformatted',
     1		access='direct',recl=iword*8)
	    write(7000+ii,rec=ncount) v(i),gl(i),gei,gii,gen,gin,
     1		-alpha1(i),beta1(i)
	    close(7000+ii)
	  enddo
c------------------------------------------------------------
	endif
c------------------------------------------------------------
	if (mod(iii,25*nstep0).eq.0) then
c------------------------------------------------------------
	  print *, 'timestep',iii,' time ',sngl(t),' Nspike ',nsptot,
     1		' Spike Rate ',sngl(nsptot/t/nmax),' Inh Rate ',
     2		sngl(isptot/t/nmax*4.D0)
	  print *,' Avgs ',sngl(condavg/nmax),
     1		sngl(curravg/nmax),sngl(condlgn/nmax),sngl(condexc/nmax),
     2		sngl(condinh/nmax),sngl(geavg/nmax),sngl(giavg/nmax)
	  print *,' Pop Rates ES/EC ',sngl(nessp/t/(0.5*0.75*nmax)),
     1		sngl(necsp/t/(0.5*0.75*nmax)),' IS/IC ',
     2		sngl(nissp/t/(0.5*0.25*nmax)),
     3		sngl(nicsp/t/(0.5*0.25*nmax))
c------------------------------------------------------------
	endif
c------------------------------------------------Output to f1
	if (mod(iii,nstep1).eq.0) then
c------------------------------------------------------------
	  ist1 = ist1 + 1
	  open(11,file=f1,status='old',form='unformatted',
     1		access='direct',recl=iword2*25*nmax)
	  write(11,rec=1) irate
	  close(11)
	  open(11,file=f1,status='old',form='unformatted',
     1		access='direct',recl=iword2)
	  write(11,rec=25*nmax+1) ist1
	  close(11)
	endif
c------------------------------------------------Output to f4
	if ((mod(iii,nstep4).eq.0).and.(t.gt.2.1D0)) then
c------------------------------------------------------------
	  ist4 = ist4 + 1
	  nc = nc + tstep4/period
	  print *,'Printing to f4, cycle-avg conds, nc = ',nc
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
	  open(14,file=f4,status='old',form='unformatted',
     1		access='direct',recl=iword*18*nmax*25)
	  write(14,rec=ist4) glgn,gexc,ginh,gtot,cond,vslave,vmem,
     1		gnexc,gninh,glgn2,gexc2,ginh2,gtot2,cond2,vslave2,
     2		vmem2,gnexc2,gninh2
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
c------------------------------------------------------------
      enddo
c------------------------------------End of TimeStepping Loop
      print *,'Neurons : ',(in(j),j=1,100)
      do ii=1,100
        open(7000+ii,status='old',form='unformatted',
     1          access='direct',recl=iword)
        write(7000+ii,rec=ncount*8+1) 1.0d0*in(ii)
        close(7000+ii)
      enddo
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
      print *,' poisson rate  = ',sngl(frtexc),'    str = ',sngl(ce0)
      print *,'      (inhib)  = ',sngl(frtinh),'    str = ',sngl(ci0)
      print *,'--------------------------------------------'
      print *,'      grating parameters : '
      print *,'   temp. freq  = ',sngl(omega/twopi),
     1		'spat. k = ',sngl(gk)
      print *,'        angle  = ',sngl(gtheta),'  phase = ',sngl(gphi)
      print *,'       tstart  = ',sngl(tstart)
      print *,'--------------------------------------------'
c------------------------------------------------------------
9000  format(1h ,75('-')/1h )
9010  format(1h ,15x,'Welcome to the Integrate-and Fire Code'/1h )
9030  format(1h ,18x,'Size of calculation ',i4,' x ',i4/1h )
9040  format(1h ,10x,'Thres  = ',e10.4,10x,' Reset = ',e10.4/
     1       1h ,10x,'Vexcit = ',e10.4,10x,'Vinhib = ',e10.4/
     2       1h ,10x,'Texcit = ',e10.4,10x,'Tinhib = ',e10.4/
     2       1h ,10x,'  Leak = ',e10.4/
ccc,10x,'     m = ',i4/
     3       1h ,10x,'   See = ',e10.4,10x,'   Sei = ',e10.4/
     4       1h ,10x,'   Sie = ',e10.4,10x,'   Sii = ',e10.4/
     5       1h ,10x,'DenExc = ',i6,14x,'AxnExc = ',i6/
     6       1h ,10x,'DenInh = ',i6,14x,'AxnInh = ',i6/1h )
9050  format(1h ,30x,'LGN Parameters'/1h )
9060  format(1h ,10x,'F-rate = ',e10.4,10x,'    g0 = ',e10.4/
     1       1h ,10x,'  tau0 = ',e10.4,10x,'  tau1 = ',e10.4/1h )
9070  format(1h ,29x,'Noise Parameters'/1h )
9080  format(1h ,10x,'FrtExc = ',e10.4,10x,'FrtInh = ',e10.4/
     1       1h ,10x,'  Cexc = ',e10.4,10x,'  Cinh = ',e10.4/1h )
c     3       1h ,10x,'   Fee = ',e10.4,10x,'   Fei = ',e10.4/
c     4       1h ,10x,'   Fie = ',e10.4,10x,'   Fii = ',e10.4/
c     5       1h ,10x,'DenExc = ',i6,14x,'AxnExc = ',i6/
c     6       1h ,10x,'DenInh = ',i6,14x,'AxnInh = ',i6/1h )
9100  format(1h ,10x,'Ntotal = ',i6,14x,'  Time = ',e10.4/
     1       1h ,10x,'Nstep1 = ',i6,14x,'   dt1 = ',e10.4/
     2       1h ,10x,'Nstep2 = ',i6,14x,'   dt2 = ',e10.4/
     3       1h ,10x,'Nstep3 = ',i6,14x,'   dt3 = ',e10.4/
     3       1h ,10x,'Nstep4 = ',i6,14x,'   dt4 = ',e10.4/1h )
c9060  format(1h ,'Nit = ',i6/
c     1       1h ,'Nd1 = ',i6/
c     2       1h ,'Nd2 = ',i6/
c     3       1h ,'Nd3 = ',i6/1h )
c------------------------------------------------------------
 9999 stop
      end
c************************************************************
       subroutine update(conduc,nspike,ispike,tspike,dt,t,iseed)
c------------------------------------------------------------
c
c  Update chain for excitatory & inhibitory after spikes
c
c------------------------------------------------------------
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
      external conduc
      data iword  / 8 /
cSGI      data iword  / 2 /
      real*4 fnsp
c------------------------------------------------------------
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
	  open(20,file='spikes.dat',status='old',form='unformatted',
     1       access='direct',recl=iword2)
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

	tt = tspike(1)
	do j=1,nspike
          call conduc(ge,se1,se2,se3,tau_e,tt,nmax)
c          call conduc(gf,sf1,sf2,sf3,taunmda,tt,nmax)
          call chain2(gf,sf1,sf2,sf3,tnrise,tndamp,tt,nmax)
          call conduc(gi,si1,si2,si3,tau_i,tt,nmax)
          call conduc(gj,sj1,sj2,sj3,tau2,tt,nmax)
	  ij = ispike(j)
c------------------------------------------------------------
ccc
ccc  calculate delta-function amplitudes given ispike(j)
ccc
	  if (excite(ij)) then
	    do i=1,nmax
	      ii = mod(nmax+i-ij,nmax) + 1
	      cnntvy = icnntvy(ii,indmap(i))*1.0d0
	      if (cnntvy.gt.0.5d0) then
	      if (excite(i)) then
		se3(i) = se3(i) + a_ee(ii) * cnntvy
		sf3(i) = sf3(i) + a_ee(ii)/tnrise*tau_e * cnntvy
	      else
		se3(i) = se3(i) + a_ie(ii) * cnntvy
		sf3(i) = sf3(i) + a_ie(ii)/tnrise*tau_e * cnntvy
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
	 	sj3(i) = sj3(i) + a_ei(ii)/tau2*tau_i * cnntvy
	      else
		si3(i) = si3(i) + a_ii(ii) * cnntvy
	 	sj3(i) = sj3(i) + a_ii(ii)/tau2*tau_i * cnntvy
	      endif
	      endif
	    enddo
	  endif
c--------------------------------------Onto next subinterval!
	  tt = tspike(j+1) - tspike(j)
c------------------------------------------------------------
	enddo
c-----------------Update chain between last spike & next time
	call conduc(ge,se1,se2,se3,tau_e,dt-tspike(nspike),nmax)
c	call conduc(gf,sf1,sf2,sf3,taunmda,dt-tspike(nspike),nmax)
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
c------------------------------------------------------------
      return
      end
c************************************************************
      subroutine e_or_i(excite,exc,iseed)
c------------------------------------------------------------
c  Set up excitatory/inhibitory tag
c  One Quarter of population is inhibitory
c    regular (or random) lattice
c------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
      logical excite(ni,nj)
      dimension exc(ni,nj)
c------------------------------------------------------------
      do j=1,nj
      do i=1,ni
        excite(i,j) = .true.
	exc(i,j) = 1.D0
c	if ((mod(j,2).eq.1).and.(mod(i,2).eq.0)) excite(i,j) = .false.
	if ((mod(j,2).eq.1).and.(mod(i,2).eq.0)) then
c
c Uncomment line below AND comment line above to make inhibitory 
c    locations random
c	if (ran2(iseed).lt.0.25D0) then
	  excite(i,j) = .false.
	  exc(i,j)    = 0.D0
	endif
      enddo
      enddo
c------------------------------------------------------------
      return
      end
c************************************************************
      subroutine visual(frtlgn0,cond0,frtexc,ce0,frtinh,ci0,t,dt,iseed)
c------------------------------------------------------------
c     Also generate noise (1-1) given firing rates
c
c     Generates LGN spike times using Poisson process 
c	with time-dependent firing rate a function of
c	visual stimulus
c------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj , nspmax = 50 )
c      parameter ( nlx = 24 , nly = 32 , nlgn = nlx*nly , maxlgn = 60 )
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
      common / chaino2/ glon,slon1,slon2,slon3,
     1		glonp,slon1p,slon2p,slon3p
      common / chainf / glf,slf1,slf2,slf3,glfp,slf1p,slf2p,slf3p
      common / chainf2/ glfn,slfn1,slfn2,slfn3,
     1		glfnp,slfn1p,slfn2p,slfn3p
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
c-------------Find firing rate as function of visual stimulus
c For CONTRAST REVERSAL, need only amplitude of sinusoid
c------------------------------------------------------------
	if ((t.gt.0.D0).and.(t.lt.tstart)) then
     	  frate = frtlgn0 + (t/tstart)*
     1		(ampson(i)*sin(omegat) + ampcon(i)*cos(omegat))
	endif
	if (t.ge.tstart) then
     	  frate = frtlgn0 + ampson(i)*sin(omegat) + ampcon(i)*cos(omegat)
	endif
c------------------------Compute spikes only if non-zero rate
	if (frate.gt.0d0) then
c------------------------------------------------------------
c Contrast Saturation
c------------------------------------------------------------
c	  ifrate = min(int(frate)*10.d0/frtlgn0,199)
c	  r2  = contr(ifrate+1)
c	  r1  = contr(ifrate)
c	  satur  = r1 + (r2-r1) * (frate*10.d0/frtlgn0-ifrate)
cc	  if (i.eq.1) print *,sngl(frate),sngl(satur),ifrate
c	  frate  = satur*frtlgn0/10.d0
c------------------------------------------------------------
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
	    glo(i)  = (glo(i)  + slo1(i)*te + slo2(i)*te2
     1              +  slo3(i)*te3)*ete
	    slo1(i) = (slo1(i) + slo2(i)*te + slo3(i)*te2)*ete
	    slo2(i) = (slo2(i) + slo3(i)*te)*ete
	    slo3(i) = (slo3(i)             )*ete + cond0

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
	    glo(i)  = (glo(i)  + slo1(i)*te + slo2(i)*te2
     1              +  slo3(i)*te3)*ete
	    slo1(i) = (slo1(i) + slo2(i)*te + slo3(i)*te2)*ete
	    slo2(i) = (slo2(i) + slo3(i)*te)*ete
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
c----------------------------------Now off-centered LGN cells
      do i=1,nlgn
	if (t.gt.pspoff(i)+0.0015) then
	if ((t.gt.0.D0).and.(t.lt.tstart)) then
     	  frate = frtlgn0 + (t/tstart)*
     1		(ampsof(i)*sin(omegat) + ampcof(i)*cos(omegat))
	endif
	if (t.ge.tstart) then
     	  frate = frtlgn0 + ampsof(i)*sin(omegat) + ampcof(i)*cos(omegat)
	endif
c------------------------------------------------------------
	if (frate.gt.0.d0) then
c------------------------------------------------------------
c Contrast Saturation
c------------------------------------------------------------
c	  ifrate = min(int(frate),199)
c	  r2  = contr(ifrate+1)
c	  r1  = contr(ifrate)
c	  satur  = r1 + (r2-r1) * (frate-ifrate)
c	  frate  = satur
c	  ifrate = min(int(frate)*10.d0/frtlgn0,199)
c	  r2  = contr(ifrate+1)
c	  r1  = contr(ifrate)
c	  satur  = r1 + (r2-r1) * (frate*10.d0/frtlgn0-ifrate)
cc	  if (i.eq.1) print *,sngl(frate),sngl(satur),ifrate
c	  frate  = satur*frtlgn0/10.d0
c------------------------------------------------------------
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
	    glf(i)  = (glf(i)  + slf1(i)*te + slf2(i)*te2
     1              +  slf3(i)*te3)*ete
	    slf1(i) = (slf1(i) + slf2(i)*te + slf3(i)*te2)*ete
	    slf2(i) = (slf2(i) + slf3(i)*te)*ete
	    slf3(i) = (slf3(i)             )*ete + cond0

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
	    glf(i)  = (glf(i)  + slf1(i)*te + slf2(i)*te2
     1              +  slf3(i)*te3)*ete
	    slf1(i) = (slf1(i) + slf2(i)*te + slf3(i)*te2)*ete
	    slf2(i) = (slf2(i) + slf3(i)*te)*ete
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
c-------------------------------------------Check noise units
      frate = frtexc
      if (frate.gt.0.0d0) then
      do i=1,nmax
	if (nlgni(i) .lt. 1) then
	tlocal = t
c------------------------------------------------------------
c  kick each input layer cell at tspike w/ delta fcn of 
c     strength Ce, Ci
c------------------------------------------------------------
	gx(i)   = gxp(i)
	sx1(i)  = sx1p(i)
	sx2(i)  = sx2p(i)
	sx3(i)  = sx3p(i)
	gy(i)   = gyp(i)
	sy1(i)  = sy1p(i)
	sy2(i)  = sy2p(i)
	sy3(i)  = sy3p(i)

 101	continue

	if (t+dt.gt.pspexc(i)) then

          dtt = pspexc(i) - tlocal

	  te  = dtt/tau_e
	  te2 = te*te/2.
	  te3 = te*te2/3.
	  ete = exp(-te)

	  tr  = dtt/tnrise
	  etr = exp(-tr)
	  td  = tspike/tndamp
	  etd = exp(-td)
	  const = tnrise/(tndamp - tnrise) * (etd - etr)

	  gx(i)  = (gx(i)  + sx1(i)*te + sx2(i)*te2
     1           +  sx3(i)*te3)*ete
	  sx1(i) = (sx1(i) + sx2(i)*te + sx3(i)*te2)*ete
	  sx2(i) = (sx2(i) + sx3(i)*te)*ete
	  sx3(i) = (sx3(i)            )*ete + ce0/tau_e

	  gy(i)  =  gy(i) * etd + const * sy3(i)
	  sy3(i) = sy3(i) * etr + ce0/tnrise

	  tlocal = pspexc(i)
c	  if (i.eq.1) print *,pspexc(i)
	  rannum = ran2(iseed)
          tnext  = -dlog(rannum)/frate
          pspexc(i) = pspexc(i) + tnext
          goto 101
        endif

	dtt = t + dt - tlocal

	te  = dtt/tau_e
	te2 = te*te/2.
	te3 = te*te2/3.
	ete = exp(-te)
	tr  = dtt/tnrise
	etr = exp(-tr)
	td  = dtt/tndamp
	etd = exp(-td)
	const = tnrise/(tndamp - tnrise) * (etd - etr)

	gx(i)  = (gx(i) + sx1(i)*te + sx2(i)*te2
     1		 +  sx3(i)*te3)*ete
	sx1(i) = (sx1(i) + sx2(i)*te + sx3(i)*te2)*ete
	sx2(i) = (sx2(i) + sx3(i)*te)*ete
	sx3(i) = (sx3(i)            )*ete

	gy(i)  =  gy(i) * etd + const * sy3(i)
	sy3(i) = sy3(i) * etr

	gxp(i)  = gx(i)
	sx1p(i) = sx1(i)
	sx2p(i) = sx2(i)
	sx3p(i) = sx3(i)
	gyp(i)  = gy(i)
	sy1p(i) = sy1(i)
	sy2p(i) = sy2(i)
	sy3p(i) = sy3(i)
	endif
      enddo
      endif
c----------------------------------Now inhibitory noise units
      frate = frtinh
      if (frate.gt.0.0d0) then
      do i=1,nmax

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
	  tj  = tspike/tau2
	  tj2 = tj*tj/2.
	  tj3 = tj*tj2/3.
	  etj = exp(-tj)

	  gn(i)  = (gn(i)  + sn1(i)*ti + sn2(i)*ti2
     1           +  sn3(i)*ti3)*eti
	  sn1(i) = (sn1(i) + sn2(i)*ti + sn3(i)*ti2)*eti
	  sn2(i) = (sn2(i) + sn3(i)*ti)*eti
	  sn3(i) = (sn3(i)            )*eti + ci0/tau_i

	  gm(i)  = (gm(i)  + sm1(i)*tj + sm2(i)*tj2
     1           +  sm3(i)*tj3)*etj
	  sm1(i) = (sm1(i) + sm2(i)*tj + sm3(i)*tj2)*etj
	  sm2(i) = (sm2(i) + sm3(i)*tj)*etj
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

	gn(i)  = (gn(i)  + sn1(i)*ti + sn2(i)*ti2
     1           +  sn3(i)*ti3)*eti
	sn1(i) = (sn1(i) + sn2(i)*ti + sn3(i)*ti2)*eti
	sn2(i) = (sn2(i) + sn3(i)*ti)*eti
	sn3(i) = (sn3(i)            )*eti
	gm(i)  = (gm(i)  + sm1(i)*tj + sm2(i)*tj2
     1           +  sm3(i)*tj3)*etj
	sm1(i) = (sm1(i) + sm2(i)*tj + sm3(i)*tj2)*etj
	sm2(i) = (sm2(i) + sm3(i)*tj)*etj
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
c------------------------------------------------------------
 999  return
      end
c************************************************************
      subroutine chain(ge,se1,se2,se3,tau,deltat,nn)
c------------------------------------------------------------
c  Oct 2000: Updating chain w/o intracortical spikes
c    PSP handled in rk2/rk4 routine
c------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
c      parameter ( ni = 128 , nj = 128 , nmax = ni*nj )
c      dimension ge(nmax),se1(nmax),se2(nmax),se3(nmax)
      dimension ge(nn),se1(nn),se2(nn),se3(nn)
c------------------------------------------------------------
      te  = deltat/tau
      te2 = te*te/2.
      te3 = te*te2/3.
      ete = exp(-te)

      do i=1,nn
c      do i=1,nmax
        ge(i)  = (ge(i) + se1(i)*te + se2(i)*te2
     1         +  se3(i)*te3)*ete
        se1(i) = (se1(i) + se2(i)*te + se3(i)*te2)*ete
        se2(i) = (se2(i) + se3(i)*te)*ete
        se3(i) = (se3(i)            )*ete
      enddo
c------------------------------------------------------------
      return
      end
c************************************************************
      subroutine chain2(ge,se1,se2,se3,trise,tdamp,deltat,nn)
c------------------------------------------------------------
c  Oct 2000: Updating difference of expon'tial w/o intracortical spikes
c------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      dimension ge(nn),se1(nn),se2(nn),se3(nn)
c------------------------------------------------------------
      tr  = deltat/trise
      etr = exp(-tr)

      td  = deltat/tdamp
      etd = exp(-td)

      const = trise/(tdamp - trise) * (etd - etr)

      do i=1,nn
	ge(i)  =  ge(i) * etd + const * se3(i)
        se3(i) = se3(i) * etr
      enddo
c------------------------------------------------------------
      return
      end
c************************************************************
      subroutine rk2(conduc,v,vnew,nspike,ispike,tspike,t,dt)
c------------------------------------------------------------
c  Oct 2000
c  uses chain to evaluate PSP (including LGN input & Noise)
c
c  Modified RK2 to solve
c    d v / dt = alpha(t) v + beta(t)
c  when v = vthres it becomes vreset
c------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
c      parameter ( nlx = 24 , nly = 32 , nlgn = nlx*nly , maxlgn = 60 )
      parameter ( nlx = 48 , nly = 64 , nlgn = nlx*nly , maxlgn = 60 )
      dimension v(nmax),vnew(nmax),gl(nmax)
      dimension alpha0(nmax),alpha1(nmax),beta0(nmax),beta1(nmax)
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
      dimension nonlgn(nmax),noflgn(nmax)
      dimension ionlgn(nmax,2*maxlgn),ioflgn(nmax,2*maxlgn)
      dimension tspike(nmax),ispike(nmax),nlgni(nmax),pspike(nmax)
c      dimension in(8),nin(8)
      dimension irate(nmax,25)
      dimension see(nmax),sei(nmax),sie(nmax),sii(nmax)
      logical excite(nmax)
c      character*80 fn(8)
c------------------------------------------------------------
      data iword / 8 /
      common / smatrx / see,sei,sie,sii
      common / spikes / irate,period
      common / lgncnd / gl
      common / vconst / vthres,vreset,vexcit,vinhib,gleak
      common / tconst / tau_e,tau_i,tau0,tau1,tau2,tnrise,tndamp
      common /  NMDA  / fnmdat,fnmdac,fgaba
      common /   rhs  / alpha1,beta1
      common / neuron / excite,nlgni
      common / lgnmap / nonlgn,noflgn,ionlgn,ioflgn
      common / chaino / glo,slo1,slo2,slo3,glop,slo1p,slo2p,slo3p
      common / chaino2/ glon,slon1,slon2,slon3,
     1		glonp,slon1p,slon2p,slon3p
      common / chainf / glf,slf1,slf2,slf3,glfp,slf1p,slf2p,slf3p
      common / chainf2/ glfn,slfn1,slfn2,slfn3,
     1		glfnp,slfn1p,slfn2p,slfn3p
      common / chaine / ge,se1,se2,se3,gep,se1p,se2p,se3p
      common / chaine2/ gf,sf1,sf2,sf3,gfp,sf1p,sf2p,sf3p
      common / chaini / gi,si1,si2,si3,gip,si1p,si2p,si3p
      common / chainj / gj,sj1,sj2,sj3,gjp,sj1p,sj2p,sj3p
      common / chainx / gx,sx1,sx2,sx3,gxp,sx1p,sx2p,sx3p
      common / chainy / gy,sy1,sy2,sy3,gyp,sy1p,sy2p,sy3p
      common / chainn / gn,sn1,sn2,sn3,gnp,sn1p,sn2p,sn3p
      common / chainm / gm,sm1,sm2,sm3,gmp,sm1p,sm2p,sm3p
      common /  avgs  / suma,sumb,suml,sumen,sumin,sume,sumi
      common /   isi  / pspike,nsptot,isptot,nessp,necsp,nissp,nicsp
c------------------------------------------------------------
      external conduc
c------------------------------------------------------------
c  First update total conductance & total current
c------------------------------------------------------------
      do i=1,nmax
	alpha0(i) = alpha1(i)
	beta0(i)  = beta1(i)
      enddo

      call conduc(glo,slo1,slo2,slo3,tau_e,dt,nlgn)
      call chain2(glon,slon1,slon2,slon3,tnrise,tndamp,dt,nlgn)
      call conduc(glf,slf1,slf2,slf3,tau_e,dt,nlgn)
      call chain2(glfn,slfn1,slfn2,slfn3,tnrise,tndamp,dt,nlgn)
      call conduc(ge,se1,se2,se3,tau_e,dt,nmax)
      call chain2(gf,sf1,sf2,sf3,tnrise,tndamp,dt,nmax)
      call conduc(gi,si1,si2,si3,tau_i,dt,nmax)
      call conduc(gj,sj1,sj2,sj3,tau2,dt,nmax)
      call conduc(gx,sx1,sx2,sx3,tau_e,dt,nmax)
      call chain2(gy,sy1,sy2,sy3,tnrise,tndamp,dt,nmax)
      call conduc(gn,sn1,sn2,sn3,tau_i,dt,nmax)
      call conduc(gm,sm1,sm2,sm3,tau2,dt,nmax)
      suma = 0.d0
      sumb = 0.d0
      suml = 0.d0
      sume = 0.d0
      sumi = 0.d0
      sumen = 0.d0
      sumin = 0.d0
      do i=1,nmax
	gl(i)	  = 0.0
	do j=1,nonlgn(i)
	  gl(i) = gl(i) + (1-fnmdat)*glo(ionlgn(i,j))
     1		+ fnmdat * glon(ionlgn(i,j))
	enddo
	do j=1,noflgn(i)
	  gl(i) = gl(i) + (1-fnmdat)*glf(ioflgn(i,j))
     1		+ fnmdat * glfn(ioflgn(i,j))
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
	alpha1(i) = -gleak - gl(i) - ginoise - gexcit - ginhib - genoise
	beta1(i)  = (gl(i)+gexcit+genoise)*vexcit + (ginoise+ginhib)*vinhib
	suma = suma - alpha1(i)
	sumb = sumb + beta1(i)
	suml = suml + gl(i)
	sume = sume + gexcit
	sumi = sumi + ginhib
	sumin = sumin + ginoise
	sumen = sumen + genoise
      enddo
c      print *,suma/nmax,sumb/nmax,suml/nmax,sume/nmax,sumi/nmax

      dt2 = 0.5d0*dt
      nspike = 0
      do i=1,nmax
	a0 = alpha0(i)
	b0 = beta0(i)
	a1 = alpha1(i)
	b1 = beta1(i)

	fk1 = a0*v(i) + b0
	fk2 = a1*(v(i)+dt*fk1) + b1
	vnew(i) = v(i) + dt2*(fk1+fk2)
c---------------------------------------------if neuron fires
	if (vnew(i).gt.vthres) then
c------------------------------------------------------------
c  Mike's modified RK2
c    1. estimate spike time (interpolate linearly)
c    2. calculate new init cond for the reset (extrapolate)
c    3. calculate vnew after spike (retake rk2 step)
c------------------------------------------------------------
	  dtsp =  2.0*(vthres-v(i))/(fk1+fk2)
	  vn   =  (vreset-dtsp*(b0+b1+b0*a1*dt)/2.)/
     1		  (1.d0+dtsp*(a0+a1+a0*a1*dt)/2.)
	  fk1  =  a0*vn + b0
	  fk2  =  a1*(vn+dt*fk1) + b1
	  vnew(i) = vn + dt2*(fk1+fk2)
ccc	  print *,i,t+dtsp,v(i),vnew(i)
	  nspike = nspike + 1
	  tspike(nspike) = dtsp
	  ispike(nspike) = i

	  nsptot = nsptot + 1
	  if ( .not. excite(i) ) isptot = isptot + 1
c------------------------------------------------------------
c  Construct histogram for spike rate 
c------------------------------------------------------------
c	  if (t.gt.1.D0-dt) then
	    tirate = mod(t+dtsp,period)
	    nirate = int(tirate*100*0.25/period) + 1
	    irate(i,nirate) = irate(i,nirate) + 1
c	  endif
c------------------------------------------endif neuron fires
	endif
c------------------------------------------------------------
      enddo
c------------------------------------------------------------
      return
      end
c************************************************************
      subroutine rk4(conduc,v,vnew,nspike,ispike,tspike,t,dt)
c------------------------------------------------------------
c     Modified RK4 to solve
c	d v / dt = alpha(t) v + beta(t)
c     when v = vthres it becomes vreset
c
c     external routine conduc calculates alpha & beta
c     spike time and neuron number is passed to external spiked
c------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
c      parameter ( nlx = 24 , nly = 32 , nlgn = nlx*nly , maxlgn = 60 )
      parameter ( nlx = 48 , nly = 64 , nlgn = nlx*nly , maxlgn = 60 )
      dimension v(nmax),vnew(nmax),gl(nmax)
      dimension alpha0(nmax),alphlf(nmax),alpha1(nmax)
      dimension beta0(nmax),bethlf(nmax),beta1(nmax)
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
      dimension tspike(nmax),ispike(nmax)
      dimension nonlgn(nmax),noflgn(nmax)
      dimension ionlgn(nmax,2*maxlgn),ioflgn(nmax,2*maxlgn)
      dimension irate(nmax,25),nlgni(nmax),pspike(nmax)
      dimension see(nmax),sei(nmax),sie(nmax),sii(nmax)
      logical excite(nmax)
c------------------------------------------------------------
      data iword / 8 /
      common / smatrx / see,sei,sie,sii
      common / spikes / irate,period
      common / lgncnd / gl
      common / vconst / vthres,vreset,vexcit,vinhib,gleak
      common / tconst / tau_e,tau_i,tau0,tau1,tau2,tnrise,tndamp
      common /  NMDA  / fnmdat,fnmdac,fgaba
      common /   rhs  / alpha1,beta1
      common / lgnmap / nonlgn,noflgn,ionlgn,ioflgn
      common / chaino / glo,slo1,slo2,slo3,glop,slo1p,slo2p,slo3p
      common / chaino2/ glon,slon1,slon2,slon3,
     1		glonp,slon1p,slon2p,slon3p
      common / chainf / glf,slf1,slf2,slf3,glfp,slf1p,slf2p,slf3p
      common / chainf2/ glfn,slfn1,slfn2,slfn3,
     1		glfnp,slfn1p,slfn2p,slfn3p
      common / chaine / ge,se1,se2,se3,gep,se1p,se2p,se3p
      common / chaine2/ gf,sf1,sf2,sf3,gfp,sf1p,sf2p,sf3p
      common / chaini / gi,si1,si2,si3,gip,si1p,si2p,si3p
      common / chainj / gj,sj1,sj2,sj3,gjp,sj1p,sj2p,sj3p
      common / chainx / gx,sx1,sx2,sx3,gxp,sx1p,sx2p,sx3p
      common / chainy / gy,sy1,sy2,sy3,gyp,sy1p,sy2p,sy3p
      common / chainn / gn,sn1,sn2,sn3,gnp,sn1p,sn2p,sn3p
      common / chainm / gm,sm1,sm2,sm3,gmp,sm1p,sm2p,sm3p
      common /  avgs  / suma,sumb,suml,sumen,sumin,sume,sumi
      common / neuron / excite,nlgni
      common /   isi  / pspike,nsptot,isptot,nessp,necsp,nissp,nicsp
c------------------------------------------------------------
      external conduc
c------------------------------------------------------------
c  First update total conductance & total current
c  Then keep i.c.'s of chain, to re-use if any neuron spikes
c------------------------------------------------------------
ccc      print *,'NMDA: ',fnmdat,fnmdac,tnrise,tndamp
      dt2    = 0.5d0*dt
      dt6    = dt/6.d0
      nspike = 0
      do i=1,nmax
	alpha0(i) = alpha1(i)
	beta0(i)  = beta1(i)
      enddo

      call conduc(glo,slo1,slo2,slo3,tau_e,dt2,nlgn)
      call chain2(glon,slon1,slon2,slon3,tnrise,tndamp,dt2,nlgn)
      call conduc(glf,slf1,slf2,slf3,tau_e,dt2,nlgn)
      call chain2(glfn,slfn1,slfn2,slfn3,tnrise,tndamp,dt2,nlgn)
      call conduc(ge,se1,se2,se3,tau_e,dt2,nmax)
      call chain2(gf,sf1,sf2,sf3,tnrise,tndamp,dt2,nmax)
      call conduc(gi,si1,si2,si3,tau_i,dt2,nmax)
      call conduc(gj,sj1,sj2,sj3,tau2,dt2,nmax)
      call conduc(gx,sx1,sx2,sx3,tau_e,dt2,nmax)
      call chain2(gy,sy1,sy2,sy3,tnrise,tndamp,dt2,nmax)
      call conduc(gn,sn1,sn2,sn3,tau_i,dt2,nmax)
      call conduc(gm,sm1,sm2,sm3,tau2,dt2,nmax)

      do i=1,nmax
	gl(i)	  = 0.0
	do j=1,nonlgn(i)
	  gl(i) = gl(i) + (1-fnmdat)*glo(ionlgn(i,j))
     1		+ fnmdat * glon(ionlgn(i,j))
	enddo
	do j=1,noflgn(i)
	  gl(i) = gl(i) + (1-fnmdat)*glf(ioflgn(i,j))
     1		+ fnmdat * glfn(ioflgn(i,j))
	enddo

	if ( excite(i) ) then
	  ginhib  = ((1-fgaba)*gi(i) + fgaba*gj(i))*sei(i)
	  gexcit  = ((1-fnmdac)*ge(i) + fnmdac*gf(i))*see(i)
	else
	  ginhib  = ((1-fgaba)*gi(i) + fgaba*gj(i))*sii(i)
	  gexcit  = ((1-fnmdac)*ge(i) + fnmdac*gf(i))*sie(i)
	endif

	ginoise   =  (1-fgaba)*gn(i) + fgaba*gm(i)
	genoise   =  (1-fnmdat)*gx(i) + fnmdat*gy(i)
	alphlf(i) = -gleak - gl(i) - gexcit - ginhib - genoise - ginoise
 	bethlf(i) = (gl(i) + gexcit + genoise)*vexcit 
     1			+ (ginoise + ginhib)*vinhib
      enddo

      call conduc(glo,slo1,slo2,slo3,tau_e,dt2,nlgn)
      call chain2(glon,slon1,slon2,slon3,tnrise,tndamp,dt2,nlgn)
      call conduc(glf,slf1,slf2,slf3,tau_e,dt2,nlgn)
      call chain2(glfn,slfn1,slfn2,slfn3,tnrise,tndamp,dt2,nlgn)
      call conduc(ge,se1,se2,se3,tau_e,dt2,nmax)
      call chain2(gf,sf1,sf2,sf3,tnrise,tndamp,dt2,nmax)
      call conduc(gi,si1,si2,si3,tau_i,dt2,nmax)
      call conduc(gj,sj1,sj2,sj3,tau2,dt2,nmax)
      call conduc(gx,sx1,sx2,sx3,tau_e,dt2,nmax)
      call chain2(gy,sy1,sy2,sy3,tnrise,tndamp,dt2,nmax)
      call conduc(gn,sn1,sn2,sn3,tau_i,dt2,nmax)
      call conduc(gm,sm1,sm2,sm3,tau2,dt2,nmax)
      suma = 0.d0
      sumb = 0.d0
      suml = 0.d0
      sume = 0.d0
      sumi = 0.d0
      sumen = 0.d0
      sumin = 0.d0
      do i=1,nmax
	gl(i)	  = 0.0
	do j=1,nonlgn(i)
	  gl(i) = gl(i) + (1-fnmdat)*glo(ionlgn(i,j))
     1		+ fnmdat * glon(ionlgn(i,j))
	enddo
	do j=1,noflgn(i)
	  gl(i) = gl(i) + (1-fnmdat)*glf(ioflgn(i,j))
     1		+ fnmdat * glfn(ioflgn(i,j))
	enddo

	if ( excite(i) ) then
	  ginhib  = ((1-fgaba)*gi(i) + fgaba*gj(i))*sei(i)
	  gexcit  = ((1-fnmdac)*ge(i) + fnmdac*gf(i))*see(i)
	else
	  ginhib  = ((1-fgaba)*gi(i) + fgaba*gj(i))*sii(i)
	  gexcit  = ((1-fnmdac)*ge(i) + fnmdac*gf(i))*sie(i)
	endif

	ginoise   =  (1-fgaba)*gn(i) + fgaba*gm(i)
	genoise   =  (1-fnmdat)*gx(i) + fnmdat*gy(i)
	alpha1(i) = -gleak - gl(i) - gexcit - ginhib - genoise - ginoise
	beta1(i)  = (gl(i) + gexcit + genoise)*vexcit 
     1			+ (ginoise + ginhib)*vinhib
	suma = suma - alpha1(i)
	sumb = sumb + beta1(i)
	suml = suml + gl(i)
	sume = sume + gexcit
	sumi = sumi + ginhib
	sumen = sumen + genoise
	sumin = sumin + ginoise
      enddo
c      print *,suma/nmax,sumb/nmax,suml/nmax,sume/nmax,sumi/nmax
c	print *,'Gm/n avg = ',gmavg/nmax,gnavg/nmax

      do i=1,nmax
	a0   =  alpha0(i)
	b0   =  beta0(i)
	ahlf =  alphlf(i)
	bhlf =  bethlf(i)
	a1   =  alpha1(i)
	b1   =  beta1(i)

	y1=dt6*(a0 + 4.d0*ahlf + a1)
	eta2    = exp(-dt2*ahlf)
	vnew(i) = exp(y1)*(v(i) + dt6*(b0 + 
     1		2.d0*bhlf*(exp(-dt2*a0)+eta2) + b1*eta2*eta2))
c	vnew(i) = exp(y1)*(v(i) + dt6*(b0 + 
c     1		2.d0*bhlf*(exp(-dt2*a0)+exp(-dt2*ahlf)) + 
c     2		b1*exp(-dt*ahlf)))
c---BTW, RK4 w/o integrating factor looks like the following:
ccc	fk1 =  a0*v(i) + b0
ccc	fk2 = ahlf*(v(i)+dt2*fk1) + bhlf
ccc	fk3 = ahlf*(v(i)+dt2*fk2) + bhlf
ccc	fk4 =  a1*(v(i)+ dt*fk3) + b1
ccc	vnew(i) = v(i) + dt*(fk1+2.D0*fk2+2.D0*fk3+fk4)/6.D0
c---------------------------------------------if neuron fires
	if (vnew(i).gt.vthres) then
c------------------------------------------------------------
c    1. estimate spike time by cubic interpolation
c    2. calculate new init cond for the reset (extrapolate)
c    3. calculate vnew after spike (retake rk4 step)
c
c    Given v0,f0,v1,f1, the cubic hermite polynomial is
c	v(t) = a + b t + c t  + d t
c    where a = v0, b = f0, c = (3*(v1-v0) - dt*(2*f0+f1))/dt/dt
c    d = (-2*(v1-v0)+dt*(f0+f1))/dt/dt/dt
c
c    6/27 Using Mike's Newton scheme plus hermite polynomial
c         also integrating factor
c------------------------------------------------------------
	  v0  =  v(i)
	  v1  =  vnew(i)
	  vt0 =  a0*v0 + b0
	  vt1 =  a1*v1 + b1
	  dtsp = dt*(vthres-v0)/(v1-v0)
	  do it=1,100
	    diff = (herm_int(dtsp,dt,v0,v1,vt0,vt1)-vthres)/
     1		hermt_int(dtsp,dt,v0,v1,vt0,vt1)
	    dtsp=dtsp-diff
	    if (abs(diff/dt).le.1.d-08) go to 333
	  enddo
	  print *,'no convergence in spiker'
          print *,i,t,dtsp,diff
	  print *

 333	  slope=hermt_int(dtsp,dt,v0,v1,vt0,vt1)
	  if ((dtsp.le.0.d0).or.(dtsp.gt.dt)
     1		.or.(slope.le.0.d0)) then
c
c       find the other roots.  get_roots returns them in incr. order.
c
	    print *,'Looking for other roots'
	    print *,i,t,t+dtsp,slope

	    call get_roots(root1,root2,iflag,dtsp,dt,
     1		v0,v1,vt0,vt1)

	    print *,dtsp,herm_int(root1,dt,v0,v1,vt0,vt1)
            print *,iflag
            print *,root1,herm_int(root1,dt,v0,v1,vt0,vt1)
            print *,root2,herm_int(root2,dt,v0,v1,vt0,vt1)

	    if (iflag.eq.1) then
c       there are two independent roots;
c       find the first root right of zero.
	      if (root1.gt.0.d0) then
		dtsp=root1
	      else
		dtsp=root2
	      end if
	    else if (iflag.eq.2) then
	      dtsp=root1
	    else if (iflag.eq.3) then
	      print *,'no extra roots found,
     1	      iflag=',iflag
	      print *,i,t,t+dtsp,v0,v1,vt0,vt1
	      ds=.01d0*dt
	      do ids=-100,200
		s=ds*ids
		vs=herm_int(s,dt,v0,v1,vt0,vt1)
		print *,t+s,vs
		vst=hermt_int(s,dt,v0,v1,vt0,vt1)
		print *,t+s,vst
	      end do
	      stop
	    end if
	  end if
c
c       2.) Calculate the new initial condition.
c
	  vnew(i) = vreset
c	  y0=0.d0
c	  ybar=herm_int(dtsp,dt,y0,y1,a0,a1)
c	  vnew(i)=vnew(i) + exp(y1-ybar)*(vreset-vthres)
	  nspike = nspike + 1
	  tspike(nspike) = dtsp
	  ispike(nspike) = i

	  nsptot = nsptot + 1
	  if (.not. excite(i) ) isptot = isptot + 1
	  pspike(i) = t + dtsp
c------------------------------------------------------------
c  Construct histogram for spike rate
c------------------------------------------------------------
	  if (t.gt.2.D0-dt) then
	    tirate = mod(t+dtsp,period)
	    nirate = int(tirate*100*0.25/period) + 1
	    irate(i,nirate) = irate(i,nirate) + 1
	  endif
c------------------------------------------endif neuron fires
	endif
c------------------------------------------------------------
c  Refractory period
c------------------------------------------------------------
ccc	if (pspike(i).gt.0.d0) then
	if (pspike(i).gt.-1.0d0) then
          if ( excite(i) ) then
            tref = 0.003
          else
            tref = 0.001
          endif
          if ((t+dt-pspike(i)).lt.tref) then
            vnew(i) = vreset
          else if ((t+dt - pspike(i)).lt.(tref+dt)) then
	    v0  =  v(i)
	    v1  =  vnew(i)
	    vt0 =  a0*v0 + b0
	    vt1 =  a1*v1 + b1
            tt = t+dt-pspike(i)-tref
            vv = herm_int(tt,dt,v0,v1,vt0,vt1)
            y0=0.d0
            ybar=herm_int(tt,dt,y0,y1,a0,a1)
            vnew(i)=vnew(i) + exp(y1-ybar)*(vreset-vv)
	  endif
        endif
c------------------------------------------------------------
      enddo
c------------------------------------------------------------
      return
      end
c************************************************************
      double precision function herm_int(s,dt,v0,v1,vt0,vt1)
c------------------------------------------------------------
c  Mike's hermite polynomial interpolant
c------------------------------------------------------------
      implicit none
      double precision s,dt,v0,v1,vt0,vt1,a,b,c,d
      double precision s2,s3,dt2,dt3
c------------------------------------------------------------
      dt2=dt*dt
      dt3=dt2*dt
      a=v0
      b=vt0
      c=(3.d0*(v1-v0)-dt*(2.d0*vt0+vt1))/dt2
      d=(-2.d0*(v1-v0)+dt*(vt0+vt1))/dt3
      s2=s*s
      s3=s2*s
      herm_int = a+b*s+c*s2+d*s3
c------------------------------------------------------------
      return
      end
c************************************************************
      double precision function hermt_int(s,dt,v0,v1,vt0,vt1)
c------------------------------------------------------------
c  Mike's hermite (derivative) polynomial interpolant
c------------------------------------------------------------
      implicit none
      double precision s,dt,v0,v1,vt0,vt1,a,b,c,d
      double precision s2,s3,dt2,dt3
c------------------------------------------------------------
      dt2=dt*dt
      dt3=dt2*dt
      a=v0
      b=vt0
      c=(3.d0*(v1-v0)-dt*(2.d0*vt0+vt1))/dt2
      d=(-2.d0*(v1-v0)+dt*(vt0+vt1))/dt3
      s2=s*s
      s3=s2*s
      hermt_int = b+2.d0*c*s+3.d0*d*s2
c------------------------------------------------------------
      return
      end
c************************************************************
      subroutine get_roots(r1,r2,iflag,r0,dt,v0,v1,vt0,vt1)
c------------------------------------------------------------
c  Mike's Newton iteration
c  This code assumes that r0.ne.0.d0.
c------------------------------------------------------------
      implicit none
      double precision dt,v0,v1,vt0,vt1,a,b,c,d
      double precision dt2,dt3,r1,r2,r0
      double precision as,bs,cs,discr,t1,t2
      integer iflag
c------------------------------------------------------------
      dt2=dt*dt
      dt3=dt2*dt
      a=v0
      b=vt0
      c=(3.d0*(v1-v0)-dt*(2.d0*vt0+vt1))/dt2
      d=(-2.d0*(v1-v0)+dt*(vt0+vt1))/dt3

      cs=d
      bs=c+r0*cs
      as=b+r0*bs
      discr=bs**2-4.d0*as*cs
      if (discr.lt.0.d0) then
c       there are no real roots.
         iflag=3
         return
      end if
      if (discr.eq.0.d0) then
c       there is one real root.
         r1=-bs/(2.d0*cs)
         iflag=2
         return
      end if
c       there are two roots.  Return in incr. order
      iflag=1
      t1=(-bs+sqrt(discr))/(2.d0*cs)
      t2=(-bs-sqrt(discr))/(2.d0*cs)
      r1=dmin1(t1,t2)
      r2=dmax1(t1,t2)
c------------------------------------------------------------
      return
      end
c************************************************************
      subroutine gaussk(prefct,bb,aa,nx,ny,al2,dx2,dy2,fglobal)
c------------------------------------------------------------
c  Oct 2000: modified for chain (do NOT fourier transform)
c
c    bb : gaussian kernel on exit
c    aa : work array
c    bb = prefct/al2 * exp(-dd/al2), dd = (i-1)*(i-1)*dx2 + (j-1)*(j-1)*dy2
c         Gaussian centered at (1,1)
c------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      dimension aa(nx*ny),bb(nx,ny)
c------------------------------------------------------------
      ni = nx/2
      nj = ny/2
      const = sqrt(dx2*dy2)/nx/ny/4.0d0/atan(1.0)

      sum = 0.0d0
      do j=1,ny
      do i=1,nx
	dd = (i-ni-1)*(i-ni-1)*dx2 + (j-nj-1)*(j-nj-1)*dy2
        bb(i,j) = const/al2*exp(-dd/al2)
ccc	bb(i,j) = 0.0D0
ccc        if (dd .lt. al2) bb(i,j) = 1.0D0
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
      print *,'Spatial kernel sum, sum*',nx,'*',ny,': ',sum,ccc

      do j=1,ny
      do i=1,nx
        jj = (j-1)*nx + i
        bb(i,j) = prefct*aa(jj)/sum
	aa(jj) = prefct/nx/ny
	bb(i,j) = (1-fglobal)*bb(i,j) + fglobal*aa(jj)
      enddo
      enddo
      print *,'gaussian ',bb(1,1),bb(nx,ny),bb(1,ny),bb(nx,1)
c------------------------------------------------------------
      return
      end
c************************************************************
      subroutine lgnrf(frtlgn)
c------------------------------------------------------------
c  Map each LGN cell to its receptive field :
c	location & spatiotemporal filter parameters
c
c  Determine each cell's firing rate given visual stimulus :
c
c  	f(t)   = f0 + \int_0^t ds \int dy G(t - s) A(x - y) I(y,s)
c	  firing rate at time t for LGN cell centered at x
c
c	where
c
c	f0     = background rate
c       G(t)   = t^5[exp(-t/t0)/t0^6 - exp(-t/t1)/t1^6]
c	A(y)   = ampa/siga/pi exp(-y^2/siga^2) 
c		- ampb/sigb/pi exp(-y^2/sigb^2) 
c	 	(overall minus one factor for off-center LGN cells)
c	I(y,s) = I_0 [ 1 + eps sin(omega s) cos( k y - phi )
c		(for contrast reversal)
c------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
c      parameter ( nlx = 24 , nly = 32 , nlgn = nlx*nly )
      parameter ( nlx = 48 , nly = 64 , nlgn = nlx*nly , maxlgn = 60 )
c------------------------------------------------------------
c     xlgn, ylgn: center of each LGN RF
c     siga, sigb: sigma of each Gaussian (RF = diff of 2 Gaussians)
c     ampa, ampb: amplitude of each Gaussian 
c     t0,   t1  : LGN kernel time constants
c     frtlgn    : background firing rate
c
c     9 Jan 2001: specialize to DG/CR for t >> t0
c	f(t) is a rectified sinusoid
c------------------------------------------------------------
      dimension xlgn(nlx*nly),ylgn(nlx*nly)
      dimension ampcon(nlx*nly),ampson(nlx*nly)
      dimension ampcof(nlx*nly),ampsof(nlx*nly)
      common / lgnpos / xlgn,ylgn
      common / lgnrfs / ampson,ampsof,ampcon,ampcof
      common / conrev / omega,gkx,gky,gphi,tstart,contrast
c------------------------------------------------------------
c  Hardwire each LGN cell
c------------------------------------------------------------
      tau0 = 0.003D0
      tau1 = 0.005D0
      ot0  = omega*tau0
      ot1  = omega*tau1
      ampc = 240.D0*((3 - 10*ot0*ot0 + 3*(ot0**4))*ot0 /
     1          (1+ot0*ot0)**6 - (3 - 10*ot1*ot1 + 3*(ot1**4))*ot1 /
     2          (1+ot1*ot1)**6)
      amps = 120.D0*((-1 + 15*ot0*ot0 - 15*(ot0**4) + ot0**6)
     1          * ot0 / (1+ot0*ot0)**6
     2          - (-1 + 15*ot1*ot1 - 15*(ot1**4) + ot1**6)
     3          * ot1 / (1+ot1*ot1)**6)
      tmpphi = atan2(amps,ampc)
      gk2    = gkx*gkx + gky*gky
c      gk2opt = 60.d0*60.d0
      gk2opt = 10.d0*10.d0
      siga2  = 1.0d0/40./40.
      sigb2  = 1.5d0*1.5d0 * siga2
      dogopt = exp(-gk2opt*siga2/4.d0) - 0.84d0*exp(-gk2opt*sigb2/4.d0)
      doggk2 = exp(-gk2   *siga2/4.d0) - 0.84d0*exp(-gk2   *sigb2/4.d0)
c------------------------------------------------------------
      ampccc = ampc/cos(tmpphi) * frtlgn/33.56D0*14.d0
      ampccc = ampccc * doggk2/dogopt * contrast
      print *,'ampccc = ',ampccc
c------------------------------------------------------------
      do i=1,nlgn
        ampson(i)  = -ampccc*cos(gphi-gkx*xlgn(i)-gky*ylgn(i))
	ampsof(i)  = -ampson(i)

        ampcon(i)  =  ampccc*sin(gphi-gkx*xlgn(i)-gky*ylgn(i))
	ampcof(i)  = -ampcon(i)
      enddo
c------------------------------------------------------------
      return
      end
c************************************************************
      subroutine lgnsatur
c------------------------------------------------------------
c
c Melinda's LGN contrast saturation function
c
c------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N) 
      dimension contr(200)
c------------------------------------------------------------
      common / consat / contr
c------------------------------------------------------------
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

c      do i=200,21,-1
c	contr(i) = contr(i-1)
c      enddo
c      do i=1,20
c	contr(i) = 0.0
c      enddo
c------------------------------------------------------------
      return
      end
c************************************************************
      subroutine genmap(icnntvy,indmap,pconnect,iseed)
c------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
c      parameter ( ni =  32 , nj =  32 , nmax = ni*nj )
c      parameter ( nlx = 32 , nly = 32 , nlgn = nlx*nly)
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
c      parameter ( nlx = 24 , nly = 32 , nlgn = nlx*nly)
      parameter ( nlx = 48 , nly = 64 , nlgn = nlx*nly)
      parameter ( nn = nmax , nmap = 32 )
c------------------------------------------------------------
c
c  Generate Postsynaptic Target Map
c
c------------------------------------------------------------
      dimension icnntvy(nmax,nmap),ind(nmax),indmap(nmax)
      data iword / 8 /
cSGI      data iword / 2 /
c------------------------------------------------------------
      nx1 = ni/4 + 1
      ny1 = nj/4 + 1
      nx2 = ni - ni/4 - 1
      ny2 = nj - nj/4 - 1

      do i=1,nmap

        do j=1,nmax
          icnntvy(j,i) = 0
        enddo

        nconn = pconnect*nmax
        print *,i,' genmap: Nconn = ',nconn

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
 19     ind(j) = ran2(iseed) * nmax
        indmap(j) = mod(ind(j),nmap) + 1
      enddo
c------------------------------------------------------------
      open(20,file='map.dat',status='new',form='unformatted',
     1  access='direct',recl=nmax*iword/2)
      print *,'Writing Map Index'
      write(20,rec=1) indmap
      print *,'Writing Connectivity Maps'
      do i=1,nmap
        write(20,rec=i+1) (icnntvy(j,i),j=1,nmax)
      enddo
      close(20)
c------------------------------------------------------------
      return
      end
c************************************************************
