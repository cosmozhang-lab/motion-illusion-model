!************************************************************
      subroutine rk2_lif(conduc,v,vnew,nspike,ispike,tspike,t,dt)
!------------------------------------------------------------
!     Modified RK2 to solve
!	d v / dt = alpha(t) v + beta(t)
!     when v = vthres it becomes vreset
!
!     external routine conduc calculates alpha & beta
!     spike time and neuron number is passed to external spiked
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
      parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
      parameter ( nlx = 48 , nly = 64 , nlgn = nlx*nly , maxlgn = 60 )
      dimension v(nmax),vnew(nmax),gl(nmax)
      dimension alpha0(nmax),alpha1(nmax)
      dimension beta0(nmax),beta1(nmax)
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
      dimension nonlgn(nmax),noflgn(nmax),nlgneff(nmax)
      dimension ionlgn(nmax,2*maxlgn),ioflgn(nmax,2*maxlgn)
      dimension irate(nmax,25),nlgni(nmax),pspike(nmax)
      dimension see(nmax),sei(nmax),sie(nmax),sii(nmax)
      logical excite(nmax)
!------------------------------------------------------------
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
      common / chaino2/ glon,slon1,slon2,slon3, & 
     		glonp,slon1p,slon2p,slon3p
      common / chainf / glf,slf1,slf2,slf3,glfp,slf1p,slf2p,slf3p
      common / chainf2/ glfn,slfn1,slfn2,slfn3, &
     		glfnp,slfn1p,slfn2p,slfn3p
      common / chaine / ge,se1,se2,se3,gep,se1p,se2p,se3p
      common / chaine2/ gf,sf1,sf2,sf3,gfp,sf1p,sf2p,sf3p
      common / chaini / gi,si1,si2,si3,gip,si1p,si2p,si3p
      common / chainj / gj,sj1,sj2,sj3,gjp,sj1p,sj2p,sj3p
      common / chainx / gx,sx1,sx2,sx3,gxp,sx1p,sx2p,sx3p
      common / chainy / gy,sy1,sy2,sy3,gyp,sy1p,sy2p,sy3p
      common / chainn / gn,sn1,sn2,sn3,gnp,sn1p,sn2p,sn3p
      common / chainm / gm,sm1,sm2,sm3,gmp,sm1p,sm2p,sm3p
      common /  avgs  / suma,sumb,suml,sumen,sumin,sume,sumi
      common / neuron / excite,nlgni,nlgneff
      common /   isi  / pspike,nsptot,isptot,nessp,necsp,nissp,nicsp
!------------------------------------------------------------
      external conduc
!------------------------------------------------------------
!  First update total conductance & total current
!  Then keep i.c.'s of chain, to re-use if any neuron spikes
!------------------------------------------------------------
      dt2 = dt/2.0d0
      nspike = 0

!      call conduc(glo,slo1,slo2,slo3,tau_e,dt,nlgn)
!      call chain2(glon,slon1,slon2,slon3,tnrise,tndamp,dt,nlgn)
!      call conduc(glf,slf1,slf2,slf3,tau_e,dt,nlgn)
!      call chain2(glfn,slfn1,slfn2,slfn3,tnrise,tndamp,dt,nlgn)
!      call conduc(ge,se1,se2,se3,tau_e,dt,nmax)
!      call chain2(gf,sf1,sf2,sf3,tnrise,tndamp,dt,nmax)
!      call conduc(gi,si1,si2,si3,tau_i,dt,nmax)
!      call conduc(gj,sj1,sj2,sj3,tau2,dt,nmax)
!      call conduc(gx,sx1,sx2,sx3,tau_e,dt,nmax)
!      call chain2(gy,sy1,sy2,sy3,tnrise,tndamp,dt,nmax)
!      call conduc(gn,sn1,sn2,sn3,tau_i,dt,nmax)
!      call conduc(gm,sm1,sm2,sm3,tau2,dt,nmax)

      do i=1,nmax
	gl(i)   = 0.0
        do j=1,nonlgn(i)
          gl(i) = gl(i) + (1-fnmdat)*glo(ionlgn(i,j)) &
                + fnmdat * glon(ionlgn(i,j))
        enddo
        do j=1,noflgn(i)
          gl(i) = gl(i) + (1-fnmdat)*glf(ioflgn(i,j)) &
                + fnmdat * glfn(ioflgn(i,j))
        enddo
	gl(i)   = gl(i) * nlgneff(i) / 30.0d0

        if ( excite(i) ) then
          ginhib  = ((1-fgaba)*gi(i) + fgaba*gj(i))*sei(i)
          gexcit  = ((1-fnmdac)*ge(i) + fnmdac*gf(i))*see(i)
        else
          ginhib  = ((1-fgaba)*gi(i) + fgaba*gj(i))*sii(i)
          gexcit  = ((1-fnmdac)*ge(i) + fnmdac*gf(i))*sie(i)
        endif

        ginoise   =  (1-fgaba)*gn(i) + fgaba*gm(i)
        genoise   =  (1-fnmdat)*gx(i) + fnmdat*gy(i)
        alpha0(i) = -gleak - gl(i) - gexcit - ginhib - genoise - ginoise
        beta0(i)  = (gl(i) + gexcit + genoise)*vexcit &
                + (ginoise + ginhib)*vinhib
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
	  gl(i) = gl(i) + (1-fnmdat)*glo(ionlgn(i,j)) &
		+ fnmdat * glon(ionlgn(i,j))
	enddo
	do j=1,noflgn(i)
	  gl(i) = gl(i) + (1-fnmdat)*glf(ioflgn(i,j)) &
		+ fnmdat * glfn(ioflgn(i,j))
	enddo
	gl(i)   = gl(i) * nlgneff(i) / 30.0d0

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
	beta1(i)  = (gl(i) + gexcit + genoise)*vexcit &
			+ (ginoise + ginhib)*vinhib
	suma = suma - alpha1(i)
	sumb = sumb + beta1(i)
	suml = suml + gl(i)
	sume = sume + gexcit
	sumi = sumi + ginhib
	sumen = sumen + genoise
	sumin = sumin + ginoise
      enddo

      do i=1,nmax
	if ( excite(i) ) then
	  tref = 0.003
	else
	  tref = 0.001
	endif

	a0   =  alpha0(i)
	b0   =  beta0(i)
	a1   =  alpha1(i)
	b1   =  beta1(i)

        fk1 = a0*v(i) + b0
        fk2 = a1*(v(i)+dt*fk1) + b1
        vnew(i) =  v(i) + dt2*(fk1+fk2)
!------------------------------------Evolve blocked potential
!        bk1 = a0*vb(i) + b0
!        bk2 = a1*(vb(i)+dt*bk1) + b1
!        vbnew(i) = vb(i) + dt2*(bk1+bk2)
!---------------------------------------------if neuron fires
	if ((vnew(i).gt.vthres) .and. ((t+dt-pspike(i)).gt.tref)) then
!------------------------------------------------------------
!    1. estimate spike time by cubic interpolation
!    2. calculate new init cond for the reset (extrapolate)
!    3. calculate vnew after spike (retake rk4 step)
!------------------------------------------------------------
	  dtsp = dt*(vthres-v(i))/(vnew(i)-v(i))
	  vnew(i) = vreset
	  nspike = nspike + 1
	  tspike(nspike) = dtsp
	  ispike(nspike) = i

	  nsptot = nsptot + 1
	  if (.not. excite(i) ) isptot = isptot + 1
	  pspike(i) = t + dtsp
!------------------------------------------------------------
!  Construct histogram for spike rate
!------------------------------------------------------------
	  if (t.gt.2.D0-dt) then
	    tirate = mod(t+dtsp,period)
	    nirate = int(tirate*100*0.25/period) + 1
	    irate(i,nirate) = irate(i,nirate) + 1
	  endif
!------------------------------------------endif neuron fires
	endif
!------------------------------------------------------------
!  Refractory period
!------------------------------------------------------------
	if (pspike(i).gt.-1.0d0) then
          if ( excite(i) ) then
            tref = 0.003
          else
            tref = 0.001
          endif
          if ((t+dt-pspike(i)).lt.tref) then
            vnew(i) = vreset
          else if ((t+dt - pspike(i)).lt.(tref+dt)) then
	    tt = t+dt-pspike(i)-tref
            vn   =  (vreset-tt*(b0+b1+b0*a1*dt)/2.)/ &
			(1.d0+dtsp*(a0+a1+a0*a1*dt)/2.)
            fk1  =  a0*vn + b0
            fk2  =  a1*(vn+dt*fk1) + b1
	    vnew(i) = vn + dt2*(fk1+fk2)
	  endif
        endif
!------------------------------------------------------------
      enddo
!------------------------------------------------------------
      return
      end
!************************************************************
