#ifndef RATX_H
#define RATX_H
#include "S3D.h"

// Contains kernels to replace the ratx function, split up to reduce
// register pressure
template <class real>
__global__ void
LAUNCH_BOUNDS (RATX_THRD, RATX_BLK)
ratx_kernel(const real* RESTRICT T, const real* RESTRICT C, real* RESTRICT RF,
        real* RESTRICT RB, const real* RESTRICT RKLOW, real TCONV)
{
    const real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;
    //const real ALOGT = LOG((TEMP));
    real CTOT = 0.0;
    real PR, PCOR, PRLOG, FCENT, FCLOG, XN;
    real CPRLOG, FLOG, FC;
    real SQR;
    const real SMALL = FLT_MIN;

    #pragma unroll 22
    for (unsigned int k=1; k<=22; k++) {
        CTOT += C(k);
    }

    real CTB_10 = CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11) + C(12)
                + (real)2.e0*C(16) + (real)2.e0*C(14) + (real)2.e0*C(15);
    real CTB_114= CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11) + C(12)
                + (real)2.e0*C(16) + (real)1.5e0*C(14) + (real)1.5e0*C(15) ;
    real CTB_16 = CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11) + C(12)
                + (real)2.e0*C(16) + (real)2.e0*C(14) + (real)2.e0*C(15) ;

    //     If fall-off (pressure correction):

    PR = RKLOW(1) * DIV(CTB_16, RF(16));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.654e-1*EXP(DIV(-TEMP,(real)9.4e1)) + (real)7.346e-1*EXP(DIV(-TEMP,(real)1.756e3))
    + EXP(DIV((real)-5.182e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(16) = RF(16) * PCOR;
    RB(16) = RB(16) * PCOR;

    PR = RKLOW(2) * DIV(CTB_10, RF(31));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)6.8e-2*EXP(DIV(-TEMP,(real)1.97e2)) + (real)9.32e-1*EXP(DIV(-TEMP,(real)1.54e3))
    + EXP(DIV((real)-1.03e4,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV (CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV (FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(31) = RF(31) * PCOR;
    RB(31) = RB(31) * PCOR;

    PR = RKLOW(3) * DIV(CTB_10, RF(39));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)4.243e-1*EXP(DIV(-TEMP,(real)2.37e2)) + (real)5.757e-1*EXP(DIV(-TEMP,(real)1.652e3))
    + EXP(DIV((real)-5.069e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(39) = RF(39) * PCOR;
    RB(39) = RB(39) * PCOR;

    PR = RKLOW(4) * DIV(CTB_10, RF(41));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.176e-1*EXP(DIV(-TEMP,(real)2.71e2)) + (real)7.824e-1*EXP(DIV(-TEMP,(real)2.755e3))
    + EXP(DIV((real)-6.57e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(41) = RF(41) * PCOR;
    RB(41) = RB(41) * PCOR;

    PR = RKLOW(5) * DIV(CTB_10, RF(48));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)3.2e-1*EXP(DIV(-TEMP,(real)7.8e1)) + (real)6.8e-1*EXP(DIV(-TEMP,(real)1.995e3))
    + EXP(DIV((real)-5.59e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(48) = RF(48) * PCOR;
    RB(48) = RB(48) * PCOR;

    PR = RKLOW(6) * DIV(CTB_10, RF(56));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)4.093e-1*EXP(DIV(-TEMP,(real)2.75e2)) + (real)5.907e-1*EXP(DIV(-TEMP,(real)1.226e3))
    + EXP(DIV((real)-5.185e3, TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(56) = RF(56) * PCOR;
    RB(56) = RB(56) * PCOR;

    PR = RKLOW(7) * DIV(CTB_10, RF(71));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.42e-1*EXP(DIV(-TEMP,(real)9.4e1)) + (real)7.58e-1*EXP(DIV(-TEMP,(real)1.555e3))
    + EXP(DIV((real)-4.2e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(71) = RF(71) * PCOR;
    RB(71) = RB(71) * PCOR;

    PR = RKLOW(8) * DIV(CTB_10, RF(78));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.17e-1*EXP(DIV(-TEMP,(real)7.4e1)) + (real)7.83e-1*EXP(DIV(-TEMP,(real)2.941e3))
    + EXP(DIV((real)-6.964e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(78) = RF(78) * PCOR;
    RB(78) = RB(78) * PCOR;

    PR = RKLOW(9) * DIV(CTB_10, RF(89));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)3.827e-1*EXP(DIV(-TEMP,(real)1.3076e1)) + (real)6.173e-1*EXP(DIV(-TEMP,(real)2.078e3))
    + EXP(DIV((real)-5.093e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(89) = RF(89) * PCOR;
    RB(89) = RB(89) * PCOR;

    PR = RKLOW(10) * DIV(CTB_10, RF(93));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)4.675e-1*EXP(DIV(-TEMP,(real)1.51e2)) + (real)5.325e-1*EXP(DIV(-TEMP,(real)1.038e3))
    + EXP(DIV((real)-4.97e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(93) = RF(93) * PCOR;
    RB(93) = RB(93) * PCOR;

    PR = RKLOW(11) * DIV(CTB_114, RF(114));
    PCOR = DIV(PR, ((real)1.0 + PR));
    RF(114) = RF(114) * PCOR;
    RB(114) = RB(114) * PCOR;

    PR = RKLOW(12) * DIV(CTB_10, RF(115));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)-9.816e-1*EXP(DIV(-TEMP,(real)5.3837e3)) +
            (real)1.9816e0*EXP(DIV(-TEMP,(real)4.2932e0))  +
            EXP(DIV((real)7.95e-2,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(115) = RF(115) * PCOR;
    RB(115) = RB(115) * PCOR;
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RATXB_THRD, RATXB_BLK)
ratxb_kernel(const real* RESTRICT T, const real* RESTRICT C, real* RESTRICT RF,
        real* RESTRICT RB, const real* RESTRICT RKLOW, real TCONV)
{
    const real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;
    //const real ALOGT = LOG((TEMP));
    real CTOT = (real)0.0;
    real PR, PCOR, PRLOG, FCENT, FCLOG, XN;
    real CPRLOG, FLOG, FC, SQR;
    const real SMALL = FLT_MIN;

    #pragma unroll 22
    for (unsigned int k=1; k<=22; k++)
    {
        CTOT += C(k);
    }

    real CTB_5  = CTOT - C(1) - C(6) + C(10) - C(12) + (real)2.e0*C(16)
                + (real)2.e0*C(14) + (real)2.e0*C(15) ;
    real CTB_9  = CTOT - (real)2.7e-1*C(1) + (real)2.65e0*C(6) + C(10) + (real)2.e0*C(16)
                + (real)2.e0*C(14) + (real)2.e0*C(15) ;
    real CTB_10 = CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11) + C(12)
                + (real)2.e0*C(16) + (real)2.e0*C(14) + (real)2.e0*C(15);
    real CTB_11 = CTOT + (real)1.4e0*C(1) + (real)1.44e1*C(6) + C(10) + (real)7.5e-1*C(11)
                + (real)2.6e0*C(12) + (real)2.e0*C(16) + (real)2.e0*C(14)
                + (real)2.e0*C(15) ;
    real CTB_12 = CTOT - C(4) - C(6) - (real)2.5e-1*C(11) + (real)5.e-1*C(12)
                + (real)5.e-1*C(16) - C(22) + (real)2.e0*C(14) + (real)2.e0*C(15) ;
    real CTB_29 = CTOT + C(1) + (real)5.e0*C(4) + (real)5.e0*C(6) + C(10)
                + (real)5.e-1*C(11) + (real)2.5e0*C(12) + (real)2.e0*C(16)
                + (real)2.e0*C(14) + (real)2.e0*C(15) ;
    real CTB_190= CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11)
                + C(12) + (real)2.e0*C(16) ;

    RF(5) = RF(5)*CTB_5*C(2)*C(2);
    RB(5) = RB(5)*CTB_5*C(1);
    RF(9) = RF(9)*CTB_9*C(2)*C(5);
    RB(9) = RB(9)*CTB_9*C(6);
    RF(10) = RF(10)*CTB_10*C(3)*C(2);
    RB(10) = RB(10)*CTB_10*C(5);
    RF(11) = RF(11)*CTB_11*C(3)*C(3);
    RB(11) = RB(11)*CTB_11*C(4);
    RF(12) = RF(12)*CTB_12*C(2)*C(4);
    RB(12) = RB(12)*CTB_12*C(7);
    RF(29) = RF(29)*CTB_29*C(11)*C(3);
    RB(29) = RB(29)*CTB_29*C(12);
    RF(46) = RF(46)*CTB_10;
    RB(46) = RB(46)*CTB_10*C(11)*C(2);
    RF(121) = RF(121)*CTOT*C(14)*C(9);
    RB(121) = RB(121)*CTOT*C(20);


    PR = RKLOW(13) * DIV(CTB_10, RF(126));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)6.63e-1*EXP(DIV(-TEMP,(real)1.707e3)) + (real)3.37e-1*EXP(DIV(-TEMP,(real)3.2e3))
    + EXP(DIV((real)-4.131e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(126) = RF(126) * PCOR;
    RB(126) = RB(126) * PCOR;

    PR = RKLOW(14) * DIV(CTB_10, RF(132));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.18e-1*EXP(DIV(-TEMP,(real)2.075e2)) + (real)7.82e-1*EXP(DIV(-TEMP,(real)2.663e3))
    + EXP(DIV((real)-6.095e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(132) = RF(132) * PCOR;
    RB(132) = RB(132) * PCOR;

    PR = RKLOW(15) * DIV(CTB_10, RF(145));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)8.25e-1*EXP(DIV(-TEMP,(real)1.3406e3)) + (real)1.75e-1*EXP(DIV(-TEMP,(real)6.e4))
    + EXP(DIV((real)-1.01398e4,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(145) = RF(145) * PCOR;
    RB(145) = RB(145) * PCOR;

    PR = RKLOW(16) * DIV(CTB_10, RF(148));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)4.5e-1*EXP(DIV(-TEMP,(real)8.9e3)) + (real)5.5e-1*EXP(DIV(-TEMP,(real)4.35e3))
    + EXP(DIV((real)-7.244e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(148) = RF(148) * PCOR;
    RB(148) = RB(148) * PCOR;

    PR = RKLOW(17) * DIV(CTB_10, RF(155));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.655e-1*EXP(DIV(-TEMP,(real)1.8e2)) + (real)7.345e-1*EXP(DIV(-TEMP,(real)1.035e3))
    + EXP(DIV((real)-5.417e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(155) = RF(155) * PCOR;
    RB(155) = RB(155) * PCOR;

    PR = RKLOW(18) * DIV(CTB_10, RF(156));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.47e-2*EXP(DIV(-TEMP,(real)2.1e2)) + (real)9.753e-1*EXP(DIV(-TEMP,(real)9.84e2))
    + EXP(DIV((real)-4.374e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(156) = RF(156) * PCOR;
    RB(156) = RB(156) * PCOR;

    PR = RKLOW(19) * DIV(CTB_10, RF(170));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)1.578e-1*EXP(DIV(-TEMP,(real)1.25e2)) + (real)8.422e-1*EXP(DIV(-TEMP,(real)2.219e3))
    + EXP(DIV((real)-6.882e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(170) = RF(170) * PCOR;
    RB(170) = RB(170) * PCOR;

    PR = RKLOW(20) * DIV(CTB_10, RF(185));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)9.8e-1*EXP(DIV(-TEMP,(real)1.0966e3)) + (real)2.e-2*EXP(DIV(-TEMP,(real)1.0966e3))
    + EXP(DIV((real)-6.8595e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(185) = RF(185) * PCOR;
    RB(185) = RB(185) * PCOR;

    PR = RKLOW(21) * DIV(CTB_190, RF(190));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)0.e0*EXP(DIV(-TEMP,(real)1.e3)) + (real)1.e0*EXP(DIV(-TEMP,(real)1.31e3))
    + EXP(DIV((real)-4.8097e4,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(190) = RF(190) * PCOR;
    RB(190) = RB(190) * PCOR;

}

template <class real>
__global__ void
LAUNCH_BOUNDS (RATX2_THRD, RATX2_BLK)
ratx2_kernel(const real* RESTRICT C, real* RESTRICT RF, real* RESTRICT RB)
{

    RF(1) = RF(1)*C(2)*C(4);
    RF(2) = RF(2)*C(3)*C(1);
    RF(3) = RF(3)*C(5)*C(1);
    RF(4) = RF(4)*C(5)*C(5);
    RF(6) = RF(6)*C(2)*C(2)*C(1);
    RF(7) = RF(7)*C(2)*C(2)*C(6);
    RF(8) = RF(8)*C(2)*C(2)*C(12);
    RF(13) = RF(13)*C(2)*C(4)*C(4);
    RF(14) = RF(14)*C(2)*C(4)*C(6);
    RF(15) = RF(15)*C(2)*C(4)*C(22);
    RF(16) = RF(16)*C(5)*C(5);
    RF(17) = RF(17)*C(7)*C(2);
    RF(18) = RF(18)*C(7)*C(2);
    RF(19) = RF(19)*C(7)*C(2);
    RF(20) = RF(20)*C(7)*C(3);
    RF(21) = RF(21)*C(7)*C(5);
    RF(22) = RF(22)*C(7)*C(7);
    RF(23) = RF(23)*C(7)*C(7);
    RF(24) = RF(24)*C(8)*C(2);
    RF(25) = RF(25)*C(8)*C(2);
    RF(26) = RF(26)*C(8)*C(3);
    RF(27) = RF(27)*C(8)*C(5);
    RF(28) = RF(28)*C(8)*C(5);
    RF(30) = RF(30)*C(11)*C(5);
    RF(31) = RF(31)*C(11)*C(1);
    RF(32) = RF(32)*C(11)*C(4);
    RF(33) = RF(33)*C(11)*C(7);
    RF(34) = RF(34)*C(3);
    RF(35) = RF(35)*C(5);
    RF(36) = RF(36)*C(1);
    RF(37) = RF(37)*C(6);
    RF(38) = RF(38)*C(4);
    RF(39) = RF(39)*C(11);
    RF(40) = RF(40)*C(12);
    RF(41) = RF(41)*C(2);
    RF(42) = RF(42)*C(2);
    RF(43) = RF(43)*C(3);
    RF(44) = RF(44)*C(3);
    RF(45) = RF(45)*C(5);
    RF(47) = RF(47)*C(4);
    RF(48) = RF(48)*C(2);
    RF(49) = RF(49)*C(1);
    RF(50) = RF(50)*C(3);
    RF(51) = RF(51)*C(4);
    RF(52) = RF(52)*C(4);
    RF(53) = RF(53)*C(5);
    RF(54) = RF(54)*C(5);
    RF(55) = RF(55)*C(7);
    RF(56) = RF(56)*C(11);
    RF(59) = RF(59)*C(22);
    RF(60) = RF(60)*C(2);
    RF(61) = RF(61)*C(3);
    RF(62) = RF(62)*C(3);
    RF(63) = RF(63)*C(5);
    RF(64) = RF(64)*C(1);
    RF(65) = RF(65)*C(4);
    RF(66) = RF(66)*C(4);
    RF(67) = RF(67)*C(6);
    RF(68) = RF(68)*C(11);
    RF(69) = RF(69)*C(12);
    RF(70) = RF(70)*C(12);
    RF(71) = RF(71)*C(13)*C(2);
    RF(72) = RF(72)*C(13)*C(2);
    RF(73) = RF(73)*C(13)*C(3);
    RF(74) = RF(74)*C(13)*C(5);
    RF(75) = RF(75)*C(13)*C(4);
    RF(76) = RF(76)*C(13)*C(7);
    RF(77) = RF(77)*C(13);
    RF(78) = RF(78)*C(9)*C(2);
    RF(79) = RF(79)*C(9)*C(3);
    RF(80) = RF(80)*C(9)*C(5);
    RF(81) = RF(81)*C(9)*C(5);
    RF(82) = RF(82)*C(9)*C(4);
    RF(83) = RF(83)*C(9)*C(4);
    RF(84) = RF(84)*C(9)*C(7);
    RF(85) = RF(85)*C(9)*C(7);
    RF(86) = RF(86)*C(9)*C(8);
    RF(87) = RF(87)*C(9);
    RF(88) = RF(88)*C(9);
    RF(89) = RF(89)*C(9);
    RF(90) = RF(90)*C(9)*C(13);
    RF(91) = RF(91)*C(9);
    RF(92) = RF(92)*C(9);
    RF(93) = RF(93)*C(9)*C(9);
    RF(94) = RF(94)*C(9)*C(9);
    RF(95) = RF(95)*C(9)*C(17);
    RF(96) = RF(96)*C(2);
    RF(97) = RF(97)*C(2);
    RF(98) = RF(98)*C(2);
    RF(99) = RF(99)*C(3);
    RF(100) = RF(100)*C(5);
    RF(101) = RF(101)*C(4);
    RF(102) = RF(102)*C(10)*C(2);
    RF(103) = RF(103)*C(10)*C(3);
    RF(104) = RF(104)*C(10)*C(5);
    RF(105) = RF(105)*C(10);
    RF(106) = RF(106)*C(10);
    RF(107) = RF(107)*C(10);
    RF(108) = RF(108)*C(17)*C(2);
    RF(109) = RF(109)*C(17)*C(3);
    RF(110) = RF(110)*C(17)*C(4);
    RF(111) = RF(111)*C(17);
    RF(112) = RF(112)*C(17);
    RF(113) = RF(113)*C(17)*C(17);
    RF(114) = RF(114)*C(14);
    RF(116) = RF(116)*C(14)*C(3);
    RF(117) = RF(117)*C(14)*C(3);
    RF(118) = RF(118)*C(14)*C(5);
    RF(119) = RF(119)*C(14)*C(5);
    RF(120) = RF(120)*C(14);
    RF(122) = RF(122)*C(2);
    RF(123) = RF(123)*C(3);
    RF(124) = RF(124)*C(5);
    RF(125) = RF(125)*C(4);
    RF(126) = RF(126)*C(18)*C(2);
    RF(127) = RF(127)*C(18)*C(2);
    RF(128) = RF(128)*C(18)*C(2);
    RF(129) = RF(129)*C(18)*C(3);
    RF(130) = RF(130)*C(18)*C(3);
    RF(131) = RF(131)*C(18)*C(5);
    RF(132) = RF(132)*C(2);
    RF(133) = RF(133)*C(2);
    RF(134) = RF(134)*C(2);
    RF(135) = RF(135)*C(3);
    RF(136) = RF(136)*C(3);
    RF(137) = RF(137)*C(5);
    RF(138) = RF(138)*C(4);
    RF(139) = RF(139)*C(4);
    RF(140) = RF(140)*C(4);
    RF(141) = RF(141)*C(7);
    RF(142) = RF(142)*C(8);
    RF(144) = RF(144)*C(9);
    RF(145) = RF(145)*C(9);
    RF(146) = RF(146)*C(9);
    RF(148) = RF(148)*C(2);
    RF(149) = RF(149)*C(2);
    RF(150) = RF(150)*C(2);
    RF(151) = RF(151)*C(3);
    RF(152) = RF(152)*C(5);
    RF(153) = RF(153)*C(4);
    RF(154) = RF(154)*C(4);
    RF(155) = RF(155)*C(15);
    RF(156) = RF(156)*C(15)*C(2);
    RF(157) = RF(157)*C(15)*C(2);
    RF(158) = RF(158)*C(15)*C(3);
    RF(159) = RF(159)*C(15)*C(3);
    RF(160) = RF(160)*C(15)*C(3);
    RF(161) = RF(161)*C(15)*C(5);
    RF(162) = RF(162)*C(15)*C(4);
    RF(163) = RF(163)*C(15)*C(7);
    RF(164) = RF(164)*C(15);
    RF(165) = RF(165)*C(15);
    RF(166) = RF(166)*C(15);
    RF(167) = RF(167)*C(15);
    RF(168) = RF(168)*C(15)*C(9);
    RF(169) = RF(169)*C(15)*C(9);
    RF(170) = RF(170)*C(2);
    RF(171) = RF(171)*C(2);
    RF(172) = RF(172)*C(3);
    RF(173) = RF(173)*C(3);
    RF(174) = RF(174)*C(4);
    RF(175) = RF(175)*C(7);
    RF(176) = RF(176)*C(7);
    RF(177) = RF(177)*C(7);
    RF(178) = RF(178)*C(8);
    RF(180) = RF(180)*C(16)*C(2);
    RF(181) = RF(181)*C(16)*C(3);
    RF(182) = RF(182)*C(16)*C(5);
    RF(183) = RF(183)*C(16);
    RF(184) = RF(184)*C(16)*C(9);
    RF(185) = RF(185)*C(20)*C(2);
    RF(186) = RF(186)*C(20)*C(2);
    RF(187) = RF(187)*C(20)*C(7);
    RF(188) = RF(188)*C(20)*C(7);
    RF(189) = RF(189)*C(20);
    RF(190) = RF(190)*C(21)*C(2);
    RF(191) = RF(191)*C(21)*C(2);
    RF(192) = RF(192)*C(21)*C(2);
    RF(193) = RF(193)*C(21)*C(3);
    RF(194) = RF(194)*C(21)*C(3);
    RF(195) = RF(195)*C(21)*C(3);
    RF(196) = RF(196)*C(21)*C(5);
    RF(197) = RF(197)*C(21)*C(7);
    RF(198) = RF(198)*C(21)*C(9);
    RF(199) = RF(199)*C(2);
    RF(200) = RF(200)*C(2);
    RF(201) = RF(201)*C(3);
    RF(202) = RF(202)*C(5);
    RF(203) = RF(203)*C(4);
    RF(204) = RF(204)*C(7);
    RF(205) = RF(205)*C(9);
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RATX4_THRD, RATX4_BLK)
ratx4_kernel(const real* RESTRICT C, real* RESTRICT RF, real* RESTRICT RB)
{
    RB(1) = RB(1)*C(3)*C(5);
    RB(2) = RB(2)*C(2)*C(5);
    RB(3) = RB(3)*C(2)*C(6);
    RB(4) = RB(4)*C(3)*C(6);
    RB(6) = RB(6)*C(1)*C(1);
    RB(7) = RB(7)*C(1)*C(6);
    RB(8) = RB(8)*C(1)*C(12);
    RB(13) = RB(13)*C(7)*C(4);
    RB(14) = RB(14)*C(7)*C(6);
    RB(15) = RB(15)*C(7)*C(22);
    RB(16) = RB(16)*C(8);
    RB(17) = RB(17)*C(3)*C(6);
    RB(18) = RB(18)*C(4)*C(1);
    RB(19) = RB(19)*C(5)*C(5);
    RB(20) = RB(20)*C(5)*C(4);
    RB(21) = RB(21)*C(4)*C(6);
    RB(22) = RB(22)*C(4)*C(8);
    RB(23) = RB(23)*C(4)*C(8);
    RB(24) = RB(24)*C(7)*C(1);
    RB(25) = RB(25)*C(5)*C(6);
    RB(26) = RB(26)*C(5)*C(7);
    RB(27) = RB(27)*C(7)*C(6);
    RB(28) = RB(28)*C(7)*C(6);
    RB(30) = RB(30)*C(12)*C(2);
    RB(31) = RB(31)*C(13);
    RB(32) = RB(32)*C(12)*C(3);
    RB(33) = RB(33)*C(12)*C(5);
    RB(34) = RB(34)*C(11)*C(2);
    RB(35) = RB(35)*C(2);
    RB(36) = RB(36)*C(2);
    RB(37) = RB(37)*C(13)*C(2);
    RB(38) = RB(38)*C(3);
    RB(39) = RB(39)*C(17);
    RB(40) = RB(40)*C(11);
    RB(41) = RB(41)*C(13);
    RB(42) = RB(42)*C(11)*C(1);
    RB(43) = RB(43)*C(11)*C(5);
    RB(44) = RB(44)*C(12)*C(2);
    RB(45) = RB(45)*C(11)*C(6);
    RB(47) = RB(47)*C(11)*C(7);
    RB(48) = RB(48)*C(9);
    RB(49) = RB(49)*C(2)*C(9);
    RB(50) = RB(50)*C(2);
    RB(51) = RB(51)*C(5);
    RB(52) = RB(52)*C(12)*C(2)*C(2);
    RB(53) = RB(53)*C(13)*C(2);
    RB(54) = RB(54)*C(6);
    RB(55) = RB(55)*C(13)*C(5);
    RB(56) = RB(56)*C(18);
    RB(57) = RB(57)*C(14)*C(2);
    RB(58) = RB(58)*C(14)*C(1);
    RB(59) = RB(59)*C(22);
    RB(60) = RB(60)*C(1);
    RB(61) = RB(61)*C(11)*C(1);
    RB(62) = RB(62)*C(2);
    RB(63) = RB(63)*C(13)*C(2);
    RB(64) = RB(64)*C(9)*C(2);
    RB(65) = RB(65)*C(2)*C(5)*C(11);
    RB(66) = RB(66)*C(11)*C(6);
    RB(67) = RB(67)*C(6);
    RB(68) = RB(68)*C(11);
    RB(69) = RB(69)*C(12);
    RB(70) = RB(70)*C(13)*C(11);
    RB(72) = RB(72)*C(1);
    RB(73) = RB(73)*C(5);
    RB(74) = RB(74)*C(6);
    RB(75) = RB(75)*C(7);
    RB(76) = RB(76)*C(8);
    RB(77) = RB(77)*C(18)*C(2);
    RB(78) = RB(78)*C(10);
    RB(79) = RB(79)*C(13)*C(2);
    RB(80) = RB(80)*C(6);
    RB(81) = RB(81)*C(6);
    RB(82) = RB(82)*C(3);
    RB(83) = RB(83)*C(5)*C(13);
    RB(84) = RB(84)*C(10)*C(4);
    RB(85) = RB(85)*C(5);
    RB(86) = RB(86)*C(10)*C(7);
    RB(87) = RB(87)*C(2);
    RB(88) = RB(88)*C(10)*C(11);
    RB(89) = RB(89)*C(19);
    RB(90) = RB(90)*C(10);
    RB(91) = RB(91)*C(15)*C(2);
    RB(92) = RB(92)*C(15)*C(2);
    RB(93) = RB(93)*C(16);
    RB(94) = RB(94)*C(2);
    RB(95) = RB(95)*C(15)*C(11);
    RB(96) = RB(96)*C(13)*C(1);
    RB(97) = RB(97)*C(9)*C(5);
    RB(98) = RB(98)*C(6);
    RB(99) = RB(99)*C(13)*C(5);
    RB(100) = RB(100)*C(13)*C(6);
    RB(101) = RB(101)*C(13)*C(7);
    RB(102) = RB(102)*C(9)*C(1);
    RB(103) = RB(103)*C(9)*C(5);
    RB(104) = RB(104)*C(9)*C(6);
    RB(105) = RB(105)*C(15)*C(2);
    RB(106) = RB(106)*C(9)*C(9);
    RB(107) = RB(107)*C(9)*C(9);
    RB(108) = RB(108)*C(11);
    RB(109) = RB(109)*C(2)*C(11)*C(11);
    RB(110) = RB(110)*C(5)*C(11)*C(11);
    RB(111) = RB(111)*C(14)*C(11);
    RB(112) = RB(112)*C(11);
    RB(113) = RB(113)*C(14)*C(11)*C(11);
    RB(115) = RB(115)*C(14)*C(2);
    RB(116) = RB(116)*C(17)*C(2);
    RB(117) = RB(117)*C(11);
    RB(118) = RB(118)*C(18)*C(2);
    RB(119) = RB(119)*C(9)*C(11);
    RB(120) = RB(120)*C(11);
    RB(122) = RB(122)*C(14)*C(2);
    RB(123) = RB(123)*C(11);
    RB(124) = RB(124)*C(18)*C(2);
    RB(125) = RB(125)*C(12);
    RB(127) = RB(127)*C(17)*C(1);
    RB(128) = RB(128)*C(9)*C(11);
    RB(129) = RB(129)*C(17)*C(5);
    RB(130) = RB(130)*C(12);
    RB(131) = RB(131)*C(17)*C(6);
    RB(132) = RB(132)*C(15);
    RB(133) = RB(133)*C(14)*C(1);
    RB(134) = RB(134)*C(1);
    RB(135) = RB(135)*C(18)*C(2);
    RB(136) = RB(136)*C(9)*C(11);
    RB(137) = RB(137)*C(14)*C(6);
    RB(138) = RB(138)*C(14)*C(7);
    RB(139) = RB(139)*C(3);
    RB(140) = RB(140)*C(13);
    RB(141) = RB(141)*C(5);
    RB(142) = RB(142)*C(15)*C(7);
    RB(143) = RB(143)*C(15)*C(11);
    RB(144) = RB(144)*C(14)*C(10);
    RB(145) = RB(145)*C(21);
    RB(146) = RB(146)*C(20)*C(2);
    RB(147) = RB(147)*C(9)*C(11);
    RB(148) = RB(148)*C(19);
    RB(149) = RB(149)*C(9);
    RB(150) = RB(150)*C(18)*C(1);
    RB(151) = RB(151)*C(18)*C(5);
    RB(152) = RB(152)*C(18)*C(6);
    RB(153) = RB(153)*C(18)*C(7);
    RB(154) = RB(154)*C(13)*C(11)*C(5);
    RB(155) = RB(155)*C(1);
    RB(157) = RB(157)*C(1);
    RB(158) = RB(158)*C(5);
    RB(159) = RB(159)*C(9);
    RB(160) = RB(160)*C(13);
    RB(161) = RB(161)*C(6);
    RB(162) = RB(162)*C(7);
    RB(163) = RB(163)*C(19)*C(5);
    RB(164) = RB(164)*C(11);
    RB(165) = RB(165)*C(20)*C(2);
    RB(166) = RB(166)*C(10);
    RB(167) = RB(167)*C(20)*C(2);
    RB(168) = RB(168)*C(10);
    RB(170) = RB(170)*C(16);
    RB(171) = RB(171)*C(15)*C(1);
    RB(172) = RB(172)*C(9)*C(13);
    RB(173) = RB(173)*C(19)*C(2);
    RB(174) = RB(174)*C(15)*C(7);
    RB(175) = RB(175)*C(16)*C(4);
    RB(176) = RB(176)*C(15)*C(8);
    RB(177) = RB(177)*C(9)*C(13)*C(5);
    RB(178) = RB(178)*C(16)*C(7);
    RB(179) = RB(179)*C(16)*C(11);
    RB(180) = RB(180)*C(1);
    RB(181) = RB(181)*C(5);
    RB(182) = RB(182)*C(6);
    RB(183) = RB(183)*C(9);
    RB(184) = RB(184)*C(10);
    RB(185) = RB(185)*C(21);
    RB(186) = RB(186)*C(10);
    RB(187) = RB(187)*C(21)*C(4);
    RB(188) = RB(188)*C(5)*C(13);
    RB(189) = RB(189)*C(21)*C(11);
    RB(191) = RB(191)*C(15)*C(9);
    RB(192) = RB(192)*C(20)*C(1);
    RB(193) = RB(193)*C(18)*C(9)*C(2);
    RB(195) = RB(195)*C(20)*C(5);
    RB(196) = RB(196)*C(20)*C(6);
    RB(197) = RB(197)*C(20)*C(8);
    RB(198) = RB(198)*C(20)*C(10);
    RB(199) = RB(199)*C(9);
    RB(200) = RB(200)*C(21)*C(1);
    RB(201) = RB(201)*C(13);
    RB(202) = RB(202)*C(21)*C(6);
    RB(203) = RB(203)*C(21)*C(7);
    RB(204) = RB(204)*C(5)*C(13);
    RB(205) = RB(205)*C(10)*C(21);
    RB(206) = RB(206)*C(20)*C(9);
}

#endif //RATX_H
