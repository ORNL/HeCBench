#include "spt.h"

void wsm ( 
        sycl::nd_item<2> &item,
        float *__restrict__ th,
  const float *__restrict__ pii,
        float *__restrict__ q,
        float *__restrict__ qc,
        float *__restrict__ qi,
        float *__restrict__ qr,
        float *__restrict__ qs,
  const float *__restrict__ den,
  const float *__restrict__ p,
  const float *__restrict__ delz,
        float *__restrict__ rain,
        float *__restrict__ rainncv,
        float *__restrict__ sr,
        float *__restrict__ snow,
        float *__restrict__ snowncv,
  const float delt,
  const int ids, const int ide,
  const int jds, const int jde,
  const int kds, const int kde,
  const int ims, const int ime,
  const int jms, const int jme,
  const int kms, const int kme,
  const int ips, const int ipe,
  const int jps, const int jpe,
  const int kps, const int kpe )
{

  float xlf, xmi, acrfac, vt2i, vt2s, supice, diameter ;
  float roqi0, xni0, qimax, value, source, factor, xlwork2 ;
  float t_k, q_k, qr_k, qc_k, qs_k, qi_k, qs1_k, qs2_k, cpm_k, xl_k, w1_k, w2_k, w3_k  ;

#define hsub   xls
#define hvap   xlv0
#define cvap   cpv
  float ttp ;
  float dldt ;
  float xa ;
  float xb ;
  float dldti ;
  float xai ;
  float xbi ;

  float qs1[MKX] ; 
  float qs2[MKX] ; 
  float rh1[MKX] ; 
  float rh2[MKX] ; 

  if ( ig < ide-ids+1 && jg < jde-jds+1 ) {

    int k ;

#include "constants.h"

    float t[MKX] ; 
    float cpm[MKX] ; 
    float xl[MKX] ; 

    for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
      t[k] = th[P3(ti,k,tj)] * pii[P3(ti,k,tj)] ;
    }

    for( k=kps-1 ;k<=kpe-1;k++) { 
      if ( qc[P3(ti,k,tj)] < 0.f ) { qc[P3(ti,k,tj)] = 0.f ; } 
      if ( qi[P3(ti,k,tj)] < 0.f ) { qi[P3(ti,k,tj)] = 0.f ; } 
      if ( qr[P3(ti,k,tj)] < 0.f ) { qr[P3(ti,k,tj)] = 0.f ; } 
      if ( qs[P3(ti,k,tj)] < 0.f ) { qs[P3(ti,k,tj)] = 0.f ; } 
    }

    //      latent heat for phase changes and heat capacity. neglect the
    //      changes during microphysical process calculation
    //      emanuel(1994)

#define CPMCAL(x) (cpd*(1.f-MAX(x,qmin))+MAX(x,qmin)*cpv)
#define XLCAL(x)  (xlv0-xlv1*((x)-t0c))

    for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
      cpm[k] = CPMCAL(q[P3(ti,k,tj)]) ;
      xl[k] = XLCAL(t[k]) ;
    }

    //      compute the minor time steps.

    float dtcldcr = 120.f ;
    int loops = delt/dtcldcr+.5f ;

    loops = MAX(loops,1) ;
    float dtcld = delt/loops ;
    if ( delt <= dtcldcr) dtcld = delt ;

    int loop ;

    for ( loop = 1 ; loop <= loops ; loop++ ) {
      //      initialize the large scale variables
      int mstep = 1 ;

      ttp=t0c+0.01f ;
      dldt=cvap-cliq ;
      xa=-dldt/rv ;
      xb=xa+hvap/(rv*ttp) ;
      dldti=cvap-cice ;
      xai=-dldti/rv ;
      xbi=xai+hsub/(rv*ttp) ;

      float tr, ltr, tt, pp, qq ;

      for ( k = kps-1 ; k <= kpe-1 ; k++ ) {

        pp = p[P3(ti,k,tj)] ;
        tt = t[k] ;
        tr = ttp/tt ;
        ltr = sycl::log(tr) ;

        qq=psat*sycl::exp(ltr*(xa)+xb*(1.f-tr)) ;
        qq=ep2*qq/(pp-qq) ;
        qs1[k] = MAX(qq,qmin) ;
        rh1[k] = MAX( q[P3(ti,k,tj)]/qs1[k],qmin) ;

        if( tt < ttp ) {
          qq=psat*sycl::exp(ltr*(xai)+xbi*(1.f-tr)) ;
        } else {
          qq=psat*sycl::exp(ltr*(xa)+xb*(1.f-tr)) ;
        }
        qq = ep2 * qq / (pp - qq) ;
        qs2[k] = MAX(qq,qmin) ;
        rh2[k] = MAX(q[P3(ti,k,tj)]/qs2[k],qmin) ;

      }

      float prevp_reg ;
      float psdep_reg ;
      float praut_reg ;
      float psaut_reg ;
      float pracw_reg ;
      float psaci_reg ;
      float psacw_reg ;
      float pigen_reg ;
      float pidep_reg ;
      float pcond_reg ;
      float psmlt_reg ;
      float psevp_reg ;
      float xni[MKX] ; 

      for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
        xni[k] = 1.e3f ;
      }

#define DIFFUS(x,y) (8.794e-5f * sycl::exp(sycl::log(x)*(1.81f)) / (y))
#define VISCOS(x,y) (1.496e-6f * ((x)*sycl::sqrt(x)) /((x)+120.f)/(y))
#define XKA(x,y) (1.414e3f*VISCOS((x),(y))*(y))
#define DIFFAC(a,b,c,d,e) ((d)*(a)*(a)/(XKA((c),(d))*rv*(c)*(c))+1.f/((e)*DIFFUS((c),(b))))
#define VENFAC(a,b,c) (sycl::exp(sycl::log((VISCOS((b),(c))/DIFFUS((b),(a))))*((.3333333f)))*sycl::rsqrt(VISCOS((b),(c)))*sycl::sqrt(sycl::sqrt(den0/(c))))

#define LAMDAR(x,y) sycl::sqrt(sycl::sqrt(pidn0r/((x)*(y))))
#define LAMDAS(x,y,z) sycl::sqrt(sycl::sqrt(pidn0s*(z)/((x)*(y))))

      // calculate mstep for this column

      float rsloper[MKX] ; 
      float rslopebr[MKX] ; 
      float rslope2r[MKX] ; 
      float rslope3r[MKX] ; 
      float rslopes[MKX] ; 
      float rslopebs[MKX] ; 
      float rslope2s[MKX] ; 
      float rslope3s[MKX] ; 
      float denfac[MKX] ; 
      float n0sfac[MKX] ; 

      float w1[MKX] ; 
      float w2[MKX] ; 
      float w3[MKX] ; 

      float w ; 
      float rmstep ;
      int numdt ;
      for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
        float supcol = t0c - t[k] ;
        n0sfac[k] = MAX(MIN(sycl::exp(alpha*supcol),n0smax/n0s),1.f) ;
        if ( qr[P3(ti,k,tj)] <= qcrmin ) {
          rsloper[k]  = rslopermax ;
          rslopebr[k] = rsloperbmax ;
          rslope2r[k] = rsloper2max ;
          rslope3r[k] = rsloper3max ;
        } else {
          rsloper[k]  = 1.f/LAMDAR(qr[P3(ti,k,tj)],den[P3(ti,k,tj)]) ;
          rslopebr[k] = sycl::exp(sycl::log(rsloper[k])*bvtr) ;
          rslope2r[k] = rsloper[k] * rsloper[k] ; 
          rslope3r[k] = rslope2r[k] * rsloper[k] ; 
        }
        if ( qs[P3(ti,k,tj)] <= qcrmin ) {
          rslopes[k]  = rslopesmax ;
          rslopebs[k] = rslopesbmax ;
          rslope2s[k] = rslopes2max ;
          rslope3s[k] = rslopes3max ;
        } else {
          rslopes[k] = 1.f/LAMDAS(qs[P3(ti,k,tj)],den[P3(ti,k,tj)],n0sfac[k]) ;
          rslopebs[k] = sycl::exp(sycl::log(rslopes[k])*bvts) ;
          rslope2s[k] = rslopes[k] * rslopes[k] ; 
          rslope3s[k] = rslope2s[k] * rslopes[k] ; 
        }
        denfac[k] = sycl::sqrt(den0/den[P3(ti,k,tj)]) ;
        w1[k] = pvtr*rslopebr[k]*denfac[k]/delz[P3(ti,k,tj)] ;
        w2[k] = pvts*rslopebs[k]*denfac[k]/delz[P3(ti,k,tj)] ;

        w = MAX(w1[k],w2[k]) ;
        numdt = MAX((int)sycl::trunc(w*dtcld+.5f+.5f),1) ;
        if ( numdt >= mstep ) mstep = numdt ;
        //-------------------------------------------------------------
        // Ni: ice crystal number concentration   [HDC 5c]
        //-------------------------------------------------------------
        float temp = (den[P3(ti,k,tj)]*MAX(qi[P3(ti,k,tj)],qmin)) ;
        temp = sycl::sqrt(sycl::sqrt(temp*temp*temp)) ;
        xni[k] = MIN(MAX(5.38e7f*temp,1.e3f),1.e6f) ;
      }
      rmstep = 1.f/mstep ;

      int n ;
      float dtcldden, coeres, rdelz ;


      float den_k, falk1_k, falk1_kp1, fall1_k, delz_k, delz_kp1 ;
      float        falk2_k, falk2_kp1, fall2_k;

      for ( n = 1 ; n <= mstep ; n++ ) {
        k = kpe - 1 ;
        den_k = den[P3(ti,k,tj)] ;
        falk1_kp1 = den_k*qr[P3(ti,k,tj)]*w1[k]*rmstep ;
        falk2_kp1 = den_k*qs[P3(ti,k,tj)]*w2[k]*rmstep ;
        dtcldden = dtcld/den_k ;
        qr[P3(ti,k,tj)] = MAX(qr[P3(ti,k,tj)]-falk1_kp1*dtcldden,0.f) ;
        qs[P3(ti,k,tj)] = MAX(qs[P3(ti,k,tj)]-falk2_kp1*dtcldden,0.f) ;
        delz_kp1 = delz[P3(ti,k,tj)] ;
        for ( k = kpe-2 ; k >= kps-1 ; k-- ) {
          den_k = den[P3(ti,k,tj)] ;
          falk1_k = den_k*qr[P3(ti,k,tj)]*w1[k]*rmstep ;
          fall1_k = falk1_k ;
          falk2_k = den_k*qs[P3(ti,k,tj)]*w2[k]*rmstep ;
          fall2_k = falk2_k ;
          dtcldden = dtcld/den_k ;
          delz_k = delz[P3(ti,k,tj)] ;
          rdelz = 1.f/delz_k ;
          qr[P3(ti,k,tj)] = MAX(qr[P3(ti,k,tj)]- (falk1_k-falk1_kp1*delz_kp1*rdelz)* dtcldden,0.f) ;
          qs[P3(ti,k,tj)] = MAX(qs[P3(ti,k,tj)]- (falk2_k-falk2_kp1*delz_kp1*rdelz)* dtcldden,0.f) ;
          delz_kp1 = delz_k ;
          falk1_kp1 = falk1_k ;
          falk2_kp1 = falk2_k ;
        }

        for ( k = kpe-1 ; k >= kps-1 ; k-- ) {
          if ( t[k] > t0c && qs[P3(ti,k,tj)] > 0.f) {
            xlf = xlf0 ;
            w3[k] = VENFAC(p[P3(ti,k,tj)],t[k],den[P3(ti,k,tj)]) ;
            coeres = rslope2s[k]*sycl::sqrt(rslopes[k]*rslopebs[2]) ;
            psmlt_reg = XKA(t[k],den[P3(ti,k,tj)])/xlf*(t0c-t[k])*pi/2.f
              *n0sfac[k]*(precs1*rslope2s[k]+precs2
                  *w3[k]*coeres) ;
            psmlt_reg = MIN(MAX(psmlt_reg*dtcld*rmstep,-qs[P3(ti,k,tj)]*rmstep),0.f) ;
            qs[P3(ti,k,tj)] += psmlt_reg ;
            qr[P3(ti,k,tj)] -= psmlt_reg ;
            t[k] += xlf/CPMCAL(q[P3(ti,k,tj)])*psmlt_reg ;
          }
        }
      }

      //---------------------------------------------------------------
      // Vice [ms-1] : fallout of ice crystal [HDC 5a]
      //---------------------------------------------------------------
      mstep = 1 ;
      numdt = 1 ;
      for ( k = kpe-1 ; k >= kps-1 ; k-- ) {
        if (qi[P3(ti,k,tj)] <= 0.f) {
          w2[k] = 0.f ;
        } else {
          xmi = den[P3(ti,k,tj)]*qi[P3(ti,k,tj)]/xni[k] ;
          diameter  = MAX(MIN(dicon * sycl::sqrt(xmi),dimax), 1.e-25f) ;
          w1[k] = 1.49e4f*sycl::exp(sycl::log(diameter)*(1.31f)) ;
          w2[k] = w1[k]/delz[P3(ti,k,tj)] ;
        }
        numdt = MAX( (int) sycl::trunc(w2[k]*dtcld+.5f+.5f),1) ;
        if(numdt > mstep) mstep = numdt ;
      }
      rmstep = 1.f/mstep ;

      float falkc_k, falkc_kp1, fallc_k = 0.f, fallc_kp1 = 0.f;
      for ( n = 1 ; n <= mstep ; n++ ) {
        k = kpe - 1 ;
        den_k = den[P3(ti,k,tj)] ;
        falkc_kp1 = den_k*qi[P3(ti,k,tj)]*w2[k]*rmstep ;
        fallc_kp1 = fallc_kp1+falkc_kp1 ;
        qi[P3(ti,k,tj)] = MAX(qi[P3(ti,k,tj)]-falkc_kp1*dtcld/den_k,0.f) ;
        delz_kp1 = delz[P3(ti,k,tj)] ;
        for ( k = kpe-2 ; k >= kps-1 ; k-- ) {
          den_k = den[P3(ti,k,tj)] ;
          falkc_k = den_k*qi[P3(ti,k,tj)]*w2[k]*rmstep ;
          fallc_k = fallc_k+falkc_k ;
          delz_k = delz[P3(ti,k,tj)] ;
          qi[P3(ti,k,tj)] = MAX(qi[P3(ti,k,tj)]-(falkc_k-falkc_kp1
                *delz_kp1/delz_k)*dtcld/den_k,0.f) ;
          delz_kp1 = delz_k ;
          falkc_kp1 = falkc_k ;
          fallc_kp1 = fallc_k ;
        }
      }
      float fallsum = fall1_k+fall2_k+fallc_k ;
      float fallsum_qsi = fall2_k+fallc_k ;

      rainncv[P2(ti,tj)] = 0.f ;
      if(fallsum > 0.f) {
        rainncv[P2(ti,tj)] = fallsum*delz[P3(ti,1,tj)]/denr*dtcld*1000.f ;
        rain[P2(ti,tj)] = fallsum*delz[P3(ti,1,tj)]/denr*dtcld*1000.f + rain[P2(ti,tj)] ;
      }
      snowncv[P2(ti,tj)] = 0.f ;
      if(fallsum_qsi > 0.f) {
        snowncv[P2(ti,tj)] = fallsum_qsi*delz[P3(ti,0,tj)]/denr*dtcld*1000.f ;
        snow[P2(ti,tj)] = fallsum_qsi*delz[P3(ti,0,tj)]/denr*dtcld*1000.f + snow[P2(ti,tj)] ;
      }
      sr[P2(ti,tj)] = 0.f ;
      if ( fallsum > 0.f ) sr[P2(ti,tj)] = fallsum_qsi*delz[P3(ti,0,tj)]/denr*dtcld*1000.f/(rainncv[P2(ti,tj)]+1.e-12f) ;

      //---------------------------------------------------------------
      // pimlt: instantaneous melting of cloud ice [HL A47] [RH83 A28]
      //       (T>T0: I->C)
      //---------------------------------------------------------------


      for ( k = kps-1 ; k <= kpe-1 ; k++ ) {

        //  note -- many of these are turned into scalars of form name_reg by _def_ above
        //  so that they will be stored in registers
        prevp_reg = 0.f ;
        psdep_reg = 0.f ;
        praut_reg = 0.f ;
        psaut_reg = 0.f ;
        pracw_reg = 0.f ;
        psaci_reg = 0.f ;
        psacw_reg = 0.f ;
        pigen_reg = 0.f ;
        pidep_reg = 0.f ;
        pcond_reg = 0.f ;
        psevp_reg = 0.f ;

        q_k =  q[P3(ti,k,tj)] ;
        t_k = t[k] ;
        qr_k =  qr[P3(ti,k,tj)] ;
        qc_k = qc[P3(ti,k,tj)] ;
        qs_k = qs[P3(ti,k,tj)] ;
        qi_k = qi[P3(ti,k,tj)] ;
        qs1_k = qs1[k] ;
        qs2_k = qs2[k] ;
        cpm_k = cpm[k] ;
        xl_k = xl[k] ;

        float supcol = t0c-t_k ;
        xlf = xls-xl_k ;
        if( supcol < 0.f ) xlf = xlf0 ;
        if( supcol < 0.f && qi_k > 0.f ) {
          qc_k = qc_k + qi_k ;
          t_k = t_k - xlf/cpm_k*qi_k ;
          qi_k = 0.f ;
        }
        //---------------------------------------------------------------
        // pihmf: homogeneous freezing of cloud water below -40c [HL A45]
        //        (T<-40C: C->I)
        //---------------------------------------------------------------
        if( supcol > 40.f && qc_k > 0.f ) {
          qi_k = qi_k + qc_k ;
          t_k = t_k + xlf/cpm_k*qc_k ;
          qc_k = 0.f ;
        }
        //---------------------------------------------------------------
        // pihtf: heterogeneous freezing of cloud water [HL A44]
        //        (T0>T>-40C: C->I)
        //---------------------------------------------------------------
        if ( supcol > 0.f && qc_k > 0.f) {
          float pfrzdtc = MIN(pfrz1*(sycl::exp(pfrz2*supcol)-1.f)
              *den[P3(ti,k,tj)]/denr/xncr*qc_k*qc_k*dtcld,qc_k) ;
          qi_k = qi_k + pfrzdtc ;
          t_k = t_k + xlf/cpm_k*pfrzdtc ;
          qc_k = qc_k-pfrzdtc ;
        }
        //---------------------------------------------------------------
        // psfrz: freezing of rain water [HL A20] [LFO 45]
        //        (T<T0, R->S)
        //---------------------------------------------------------------
        if( supcol > 0.f && qr_k > 0.f ) {
          float temp = rsloper[k] ;
          temp = temp*temp*temp*temp*temp*temp*temp ;
          float pfrzdtr = MIN(20.f*(pi*pi)*pfrz1*n0r*denr/den[P3(ti,k,tj)]
              *(sycl::exp(pfrz2*supcol)-1.f)*temp*dtcld,
              qr_k) ;
          qs_k = qs_k + pfrzdtr ;
          t_k = t_k + xlf/cpm_k*pfrzdtr ;
          qr_k = qr_k-pfrzdtr ;
        }

        //----------------------------------------------------------------
        //     rsloper: reverse of the slope parameter of the rain(m)
        //     xka:    thermal conductivity of air(jm-1s-1k-1)
        //     work1:  the thermodynamic term in the denominator associated with
        //             heat conduction and vapor diffusion
        //             (ry88, y93, h85)
        //     work2: parameter associated with the ventilation effects(y93)

        n0sfac[k] = MAX(MIN(sycl::exp(alpha*supcol),n0smax/n0s),1.f) ;
        if ( qr_k <= qcrmin ) {
          rsloper[k]  = rslopermax ;
          rslopebr[k] = rsloperbmax ;
          rslope2r[k] = rsloper2max ;
          rslope3r[k] = rsloper3max ;
        } else {
          rsloper[k] = 1.f/(sycl::sqrt(sycl::sqrt(pidn0r/((qr_k)*(den[P3(ti,k,tj)]))))) ;
          rslopebr[k] = sycl::exp(sycl::log(rsloper[k])*bvtr) ;
          rslope2r[k] = rsloper[k] * rsloper[k] ;
          rslope3r[k] = rslope2r[k] * rsloper[k] ;
        }
        if ( qs_k <= qcrmin ) {
          rslopes[k]  = rslopesmax ;
          rslopebs[k] = rslopesbmax ;
          rslope2s[k] = rslopes2max ;
          rslope3s[k] = rslopes3max ;
        } else {
          rslopes[k] = 1.f/(sycl::sqrt(sycl::sqrt(pidn0s*(n0sfac[k])/((qs_k)*(den[P3(ti,k,tj)]))))) ;
          rslopebs[k] = sycl::exp(sycl::log(rslopes[k])*bvts) ;
          rslope2s[k] = rslopes[k] * rslopes[k] ;
          rslope3s[k] = rslope2s[k] * rslopes[k] ;
        }

        w1_k = DIFFAC(xl_k,p[P3(ti,k,tj)],t_k,den[P3(ti,k,tj)],qs1_k) ;
        w2_k = DIFFAC(xls,p[P3(ti,k,tj)],t_k,den[P3(ti,k,tj)],qs2_k) ;
        w3_k = VENFAC(p[P3(ti,k,tj)],t_k,den[P3(ti,k,tj)]) ;

        //
        //===============================================================
        //
        // warm rain processes
        //
        // - follows the processes in RH83 and LFO except for autoconcersion
        //
        //===============================================================
        //
        float supsat = MAX(q_k,qmin)-qs1_k ;
        float satdt = supsat/dtcld ;
        //---------------------------------------------------------------
        // praut: auto conversion rate from cloud to rain [HDC 16]
        //        (C->R)
        //---------------------------------------------------------------
        if(qc_k > qc0) {
          praut_reg = qck1*sycl::exp(sycl::log(qc_k)*((7.f/3.f))) ;
          praut_reg = MIN(praut_reg,qc_k/dtcld) ;
        }
        //---------------------------------------------------------------
        // pracw: accretion of cloud water by rain [HL A40] [LFO 51]
        //        (C->R)
        //---------------------------------------------------------------
        if(qr_k > qcrmin && qc_k > qmin) {
          pracw_reg = MIN(pacrr*rslope3r[k]*rslopebr[k]
              *qc_k*denfac[k],qc_k/dtcld) ;
        }
        //---------------------------------------------------------------
        // prevp: evaporation/condensation rate of rain [HDC 14]
        //        (V->R or R->V)
        //---------------------------------------------------------------
        if(qr_k > 0.f) {
          coeres = rslope2r[k]*sycl::sqrt(rsloper[k]*rslopebr[k]) ;
          prevp_reg = (rh1[k]-1.f)*(precr1*rslope2r[k]
              +precr2*w3_k*coeres)/w1_k ;
          if(prevp_reg < 0.f) {
            prevp_reg = MAX(prevp_reg,-qr_k/dtcld) ;
            prevp_reg = MAX(prevp_reg,satdt/2.f) ;
          } else {
            prevp_reg = MIN(prevp_reg,satdt/2.f) ;
          }
        }

        //
        //===============================================================
        //
        // cold rain processes
        //
        // - follows the revised ice microphysics processes in HDC
        // - the processes same as in RH83 and RH84  and LFO behave
        //   following ice crystal hapits defined in HDC, inclduing
        //   intercept parameter for snow (n0s), ice crystal number
        //   concentration (ni), ice nuclei number concentration
        //   (n0i), ice diameter (d)
        //
        //===============================================================
        //
        float rdtcld = 1.f/dtcld ;
        supsat = MAX(q_k,qmin)-qs2_k ;
        satdt = supsat/dtcld ;
        int ifsat = 0 ;
        //-------------------------------------------------------------
        // Ni: ice crystal number concentraiton   [HDC 5c]
        //-------------------------------------------------------------
        float temp = (den[P3(ti,k,tj)]*MAX(qi_k,qmin)) ;
        temp = sycl::sqrt(sycl::sqrt(temp*temp*temp)) ;
        xni[k] = MIN(MAX(5.38e7f*temp,1.e3f),1.e6f) ;
        float eacrs = sycl::exp(0.07f*(-supcol)) ;
        //-------------------------------------------------------------
        // psacw: Accretion of cloud water by snow  [HL A7] [LFO 24]
        //        (T<T0: C->S, and T>=T0: C->R)
        //-------------------------------------------------------------
        if(qs_k > qcrmin && qc_k > qmin) {
          psacw_reg = MIN(pacrc*n0sfac[k]*rslope3s[k] 
              *rslopebs[k]*qc_k*denfac[k]
              ,qc_k*rdtcld) ;
        }
        //
        if(supcol > 0) {
          if(qs_k > qcrmin && qi_k > qmin) {
            xmi = den[P3(ti,k,tj)]*qi_k/xni[k] ;
            diameter  = MIN(dicon * sycl::sqrt(xmi),dimax) ;
            vt2i = 1.49e4f*sycl::pow(diameter,1.31f) ;
            vt2s = pvts*rslopebs[k]*denfac[k] ;
            //-------------------------------------------------------------
            // psaci: Accretion of cloud ice by rain [HDC 10]
            //        (T<T0: I->S)
            //-------------------------------------------------------------
            acrfac = 2.f*rslope3s[k]+2.f*diameter*rslope2s[k]
              +diameter*diameter*rslopes[k] ;
            psaci_reg = pi*qi_k*eacrs*n0s*n0sfac[k]
              *sycl::fabs(vt2s-vt2i)*acrfac*.25f ;
          }
          //-------------------------------------------------------------
          // pidep: Deposition/Sublimation rate of ice [HDC 9]
          //       (T<T0: V->I or I->V)
          //-------------------------------------------------------------
          if(qi_k > 0 && ifsat != 1) {
            xmi = den[P3(ti,k,tj)]*qi_k/xni[k] ;
            diameter = dicon * sycl::sqrt(xmi) ;
            pidep_reg = 4.f*diameter*xni[k]*(rh2[k]-1.f)/w2_k ;
            supice = satdt-prevp_reg ;
            if(pidep_reg < 0.f) {
              pidep_reg = MAX(MAX(pidep_reg,satdt*.5f),supice) ;
              pidep_reg = MAX(pidep_reg,-qi_k*rdtcld) ;
            } else {
              pidep_reg = MIN(MIN(pidep_reg,satdt*.5f),supice) ;
            }
            if(sycl::fabs(prevp_reg+pidep_reg) >= sycl::fabs(satdt)) ifsat = 1 ;
          }
          //-------------------------------------------------------------
          // psdep: deposition/sublimation rate of snow [HDC 14]
          //        (V->S or S->V)
          //-------------------------------------------------------------
          if( qs_k > 0.f && ifsat != 1) {
            coeres = rslope2s[k]*sycl::sqrt(rslopes[k]*rslopebs[k]) ;
            psdep_reg = (rh2[k]-1.f)*n0sfac[k]
              *(precs1*rslope2s[k]+precs2
                  *w3_k*coeres)/w2_k ;
            supice = satdt-prevp_reg-pidep_reg ;
            if(psdep_reg < 0.f) {
              psdep_reg = MAX(psdep_reg,-qs_k*rdtcld) ;
              psdep_reg = MAX(MAX(psdep_reg,satdt*.5f),supice) ;
            } else {
              psdep_reg = MIN(MIN(psdep_reg,satdt*.5f),supice) ;
            }
            if(sycl::fabs(prevp_reg+pidep_reg+psdep_reg) >= sycl::fabs(satdt))
              ifsat = 1 ;
          }
          //-------------------------------------------------------------
          // pigen: generation(nucleation) of ice from vapor [HL A50] [HDC 7-8]
          //       (T<T0: V->I)
          //-------------------------------------------------------------
          if(supsat > 0 && ifsat != 1) {
            supice = satdt-prevp_reg-pidep_reg-psdep_reg ; 
            xni0 = 1.e3f*sycl::exp(0.1f*supcol) ;
            roqi0 = 4.92e-11f*sycl::exp(sycl::log(xni0)*(1.33f));
            pigen_reg = MAX(0.f,(roqi0/den[P3(ti,k,tj)]-MAX(qi_k,0.f))
                *rdtcld) ;
            pigen_reg = MIN(MIN(pigen_reg,satdt),supice) ;
          }
          //
          //-------------------------------------------------------------
          // psaut: conversion(aggregation) of ice to snow [HDC 12]
          //       (T<T0: I->S)
          //-------------------------------------------------------------
          if(qi_k > 0.f) {
            qimax = roqimax/den[P3(ti,k,tj)] ;
            psaut_reg = MAX(0.f,(qi_k-qimax)*rdtcld) ;
          }
        }
        //-------------------------------------------------------------
        // psevp: Evaporation of melting snow [HL A35] [RH83 A27]
        //       (T>T0: S->V)
        //-------------------------------------------------------------
        if(supcol < 0.f) {
          if(qs_k > 0.f && rh1[k] < 1.f) {
            psevp_reg = psdep_reg*w2_k/w1_k ;
          }  // asked Jimy about this, 11.6.07, JM
          psevp_reg = MIN(MAX(psevp_reg,-qs_k*rdtcld),0.f) ;
        }


        //     check mass conservation of generation terms and feedback to the
        //     large scale
        if(t_k<=t0c) {
          //
          //     cloud water
          //
          value = MAX(qmin,qc_k) ;
          source = (praut_reg+pracw_reg+psacw_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            praut_reg = praut_reg*factor ;
            pracw_reg = pracw_reg*factor ;
            psacw_reg = psacw_reg*factor ;
          }
          //
          //     cloud ice
          //
          value = MAX(qmin,qi_k) ;
          source = (psaut_reg+psaci_reg-pigen_reg-pidep_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            psaut_reg = psaut_reg*factor ;
            psaci_reg = psaci_reg*factor ;
            pigen_reg = pigen_reg*factor ;
            pidep_reg = pidep_reg*factor ;
          }

          //
          //     rain (added for WRFV3.0.1)
          //
          value = MAX(qmin,qr_k) ;
          source = (-praut_reg+pracw_reg-prevp_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            praut_reg = praut_reg*factor ;
            pracw_reg = pracw_reg*factor ;
            prevp_reg = prevp_reg*factor ;
          }
          //
          //     snow (added for WRFV3.0.1)
          //
          value = MAX(qmin,qs_k) ;
          source = (-psdep_reg+psaut_reg-psaci_reg-psacw_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            psdep_reg = psdep_reg*factor ;
            psaut_reg = psaut_reg*factor ;
            psaci_reg = psaci_reg*factor ;
            psacw_reg = psacw_reg*factor ;
          }
          //     (end added for WRFV3.0.1)

          //
          w3_k=-(prevp_reg+psdep_reg+pigen_reg+pidep_reg) ;
          //     update
          q_k = q_k+w3_k*dtcld ;
          qc_k = MAX(qc_k-(praut_reg+pracw_reg+psacw_reg)*dtcld,0.f) ;
          qr_k = MAX(qr_k+(praut_reg+pracw_reg+prevp_reg)*dtcld,0.f) ;
          qi_k = MAX(qi_k-(psaut_reg+psaci_reg-pigen_reg-pidep_reg)*dtcld,0.f) ;
          qs_k = MAX(qs_k+(psdep_reg+psaut_reg+psaci_reg+psacw_reg)*dtcld,0.f) ;
          xlf = xls-xl_k ;
          xlwork2 = -xls*(psdep_reg+pidep_reg+pigen_reg)-xl_k*prevp_reg-xlf*psacw_reg ;
          t_k = t_k-xlwork2/cpm_k*dtcld ;
        } else {
          //
          //     cloud water
          //
          value = MAX(qmin,qc_k) ;
          source=(praut_reg+pracw_reg+psacw_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            praut_reg = praut_reg*factor ;
            pracw_reg = pracw_reg*factor ;
            psacw_reg = psacw_reg*factor ;
          }
          //
          //     rain (added for WRFV3.0.1)
          //
          value = MAX(qmin,qr_k) ;
          source = (-praut_reg-pracw_reg-prevp_reg-psacw_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            praut_reg = praut_reg*factor ;
            pracw_reg = pracw_reg*factor ;
            prevp_reg = prevp_reg*factor ;
            psacw_reg = psacw_reg*factor ;
          }
          //     (end added for WRFV3.0.1)
          //
          //     snow
          //
          value = MAX(qcrmin,qs_k) ;
          source=(-psevp_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            psevp_reg = psevp_reg*factor ;
          }
          w3_k=-(prevp_reg+psevp_reg) ;
          //     update
          q_k = q_k+w3_k*dtcld ;
          qc_k = MAX(qc_k-(praut_reg+pracw_reg+psacw_reg)*dtcld,0.f) ;
          qr_k = MAX(qr_k+(praut_reg+pracw_reg+prevp_reg +psacw_reg)*dtcld,0.f) ;
          qs_k = MAX(qs_k+psevp_reg*dtcld,0.f) ;
          xlf = xls-xl_k ;
          xlwork2 = -xl_k*(prevp_reg+psevp_reg) ;
          t_k = t_k-xlwork2/cpm_k*dtcld ;
        }
        //
        // Inline expansion for fpvs
        cvap = cpv ;
        ttp=t0c+0.01f ;
        dldt=cvap-cliq ;
        xa=-dldt/rv ;
        xb=xa+hvap/(rv*ttp) ;
        dldti=cvap-cice ;
        xai=-dldti/rv ;
        xbi=xai+hsub/(rv*ttp) ;
        tr=ttp/t_k ;
        qs1_k=psat*sycl::exp(sycl::log(tr)*(xa))*sycl::exp(xb*(1.f-tr)) ;
        qs1_k = ep2 * qs1_k / (p[P3(ti,k,tj)] - qs1_k) ;
        qs1_k = MAX(qs1_k,qmin) ;

        //  pcond: condensational/evaporational rate of cloud water [HL A46] [RH83 A6]
        //     if there exists additional water vapor condensated/if
        //     evaporation of cloud water is not enough to remove subsaturation
        
        w1_k = ((MAX(q_k,qmin)-(qs1_k)) /
            (1.f+(xl_k)*(xl_k)/(rv*(cpm_k))*(qs1_k)/((t_k)*(t_k)))) ;
        // w3_k = qc_k+w1_k ;   NOT USED
        pcond_reg = MIN(MAX(w1_k/dtcld,0.f),MAX(q_k,0.f)/dtcld) ;
        if(qc_k > 0.f && w1_k < 0.f) {
          pcond_reg = MAX(w1_k,-qc_k)/dtcld ;
        }
        q_k = q_k-pcond_reg*dtcld ;
        qc_k = MAX(qc_k+pcond_reg*dtcld,0.f) ;
        t_k = t_k+pcond_reg*xl_k/cpm_k*dtcld ;

        //     padding for small values
        if(qc_k <= qmin) qc_k = 0.f ;
        if(qi_k <= qmin) qi_k = 0.f ;

        q[P3(ti,k,tj)] = q_k ;
        t[k] = t_k ;
        qr[P3(ti,k,tj)] = qr_k ;
        qc[P3(ti,k,tj)] = qc_k ;
        qs[P3(ti,k,tj)] = qs_k ;
        qi[P3(ti,k,tj)] = qi_k ;
        qs1[k] = qs1_k ;

      }
    }

    for ( k = kps-1 ; k <= kpe-1 ; k++ )
      th[P3(ti,k,tj)] = t[k] / pii[P3(ti,k,tj)] ;
  }
}
