#ifndef RATT_H
#define RATT_H

#include "S3D.h"

template <class real>
__global__ void
LAUNCH_BOUNDS (RATT_THRD, RATT_BLK)
ratt_kernel(const real* RESTRICT T, real* RESTRICT RF, real TCONV)
{

    const real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;

    const real ALOGT = LOG(TEMP);
    const real TI = (real)1.0e0/(TEMP);
    const real TI2 = TI*TI;
    real TMP;

    RF(1) = EXP((real)3.20498617e1 -(real)7.25286183e3*TI);
    RF(2) = EXP((real)1.08197783e1 +(real)2.67e0*ALOGT - (real)3.16523284e3*TI);
    RF(3) = EXP((real)1.9190789e1  +(real)1.51e0*ALOGT - (real)1.72603317e3*TI);
    RF(4) = EXP((real)1.0482906e1  +(real)2.4e0*ALOGT + (real)1.06178717e3*TI);
    RF(5) = (real)1.e18*TI;
    RF(6) = EXP((real)3.90385861e1 -(real)6.e-1*ALOGT);
    RF(7) = EXP((real)4.55408762e1 -(real)1.25e0*ALOGT);
    RF(8) = (real)5.5e20*TI2;
    RF(9) = (real)2.2e22*TI2;
    RF(10) = (real)5.e17*TI;
    RF(11) = (real)1.2e17*TI;
    RF(12) = EXP((real)4.24761511e1 - (real)8.6e-1*ALOGT);
    RF(13) = EXP((real)4.71503141e1 - (real)1.72e0*ALOGT);
    RF(14) = EXP((real)4.42511034e1 - (real)7.6e-1*ALOGT);
    RF(15) = EXP((real)4.47046282e1 - (real)1.24e0*ALOGT);
    RF(16) = EXP((real)3.19350862e1 - (real)3.7e-1*ALOGT);
    RF(17) = EXP((real)2.90097872e1 - (real)3.37658384e2*TI);
    RF(18) = EXP((real)3.04404238e1 - (real)4.12637667e2*TI);
    RF(19) = EXP((real)3.18908801e1 - (real)1.50965e2*TI);
    RF(20) = (real)2.e13;
    RF(21) = EXP((real)3.14683206e1 + (real)2.51608334e2*TI);
    RF(22) = EXP((real)2.55908003e1 + (real)8.20243168e2*TI);
    RF(23) = EXP((real)3.36712758e1 - (real)6.03860001e3*TI);
    RF(24) = EXP((real)1.6308716e1  + (real)2.e0*ALOGT - (real)2.61672667e3*TI);
    RF(25) = EXP((real)2.99336062e1 - (real)1.81158e3*TI);
    RF(26) = EXP((real)1.60803938e1 + (real)2.e0*ALOGT - (real)2.01286667e3*TI);
    RF(27) = EXP((real)2.81906369e1 - (real)1.61029334e2*TI);
    RF(28) = EXP((real)3.39940492e1 - (real)4.81075134e3*TI);
    RF(29) = EXP((real)3.40312786e1 - (real)1.50965e3*TI);
    RF(30) = EXP((real)1.76783433e1 + (real)1.228e0*ALOGT - (real)3.52251667e1*TI);
    RF(31) = EXP((real)1.75767107e1 + (real)1.5e0*ALOGT - (real)4.00560467e4*TI);
    RF(32) = EXP((real)2.85473118e1 - (real)2.40537567e4*TI);
    RF(33) = EXP((real)3.26416564e1 - (real)1.18759134e4*TI);
    RF(34) = (real)5.7e13;
    RF(35) = (real)3.e13;
    RF(36) = EXP((real)1.85223344e1 + (real)1.79e0*ALOGT - (real)8.40371835e2*TI);
    RF(37) = EXP((real)2.93732401e1 + (real)3.79928584e2*TI);
    RF(38) = (real)3.3e13;
    RF(39) = (real)5.e13;
    RF(40) = EXP((real)2.88547965e1 - (real)3.47219501e2*TI);
    RF(41) = EXP((real)2.77171988e1 + (real)4.8e-1*ALOGT + (real)1.30836334e2*TI);
    RF(42) = (real)7.34e13;
    RF(43) = (real)3.e13;
    RF(44) = (real)3.e13;
    RF(45) = (real)5.e13;
    RF(46) = EXP((real)3.9769885e1  - (real)1.e0*ALOGT - (real)8.55468335e3*TI);
    RF(47) = EXP((real)2.96591694e1 - (real)2.01286667e2*TI);
    RF(48) = EXP((real)3.77576522e1 - (real)8.e-1*ALOGT);
    RF(49) = EXP((real)1.31223634e1 + (real)2.e0*ALOGT - (real)3.63825651e3*TI);
    RF(50) = (real)8.e13;
    TMP = EXP((real)-7.54825001e2*TI);
    RF(51) = (real)1.056e13 * TMP;
    RF(52) = (real)2.64e12 * TMP;
    RF(53) = (real)2.e13;
    RF(54) = EXP((real)1.62403133e1 +(real)2.e0*ALOGT -(real)1.50965e3*TI);
    RF(55) = (real)2.e13;
    RF(56) = EXP((real)2.74203001e1 +(real)5.e-1*ALOGT -(real)2.26950717e3*TI);
    RF(57) = (real)4.e13;
    RF(58) = (real)3.2e13;
    RF(59) = EXP((real)3.03390713e1 -(real)3.01930001e2*TI);
    RF(60) = (real)3.e13;
    RF(61) = (real)1.5e13;
    RF(62) = (real)1.5e13;
    RF(63) = (real)3.e13;
    RF(64) = (real)7.e13;
    RF(65) = (real)2.8e13;
    RF(66) = (real)1.2e13;
    RF(67) = (real)3.e13;
    RF(68) = (real)9.e12;
    RF(69) = (real)7.e12;
    RF(70) = (real)1.4e13;
    RF(71) = EXP((real)2.7014835e1  + (real)4.54e-1*ALOGT - (real)1.30836334e3*TI);
    RF(72) = EXP((real)2.38587601e1 + (real)1.05e0*ALOGT - (real)1.64803459e3*TI);
    RF(73) = EXP((real)3.12945828e1 - (real)1.781387e3*TI);
    RF(74) = EXP((real)2.19558261e1 + (real)1.18e0*ALOGT + (real)2.2493785e2*TI);
    RF(75) = EXP((real)3.22361913e1 - (real)2.01286667e4*TI);
    TMP = EXP( (real)-4.02573334e3*TI);
    RF(76) = (real)1.e12 * TMP;
    RF(127) = (real)5.e13 * TMP;
    RF(129) = (real)1.e13 * TMP;
    RF(77) = EXP((real)3.21806786e1 + (real)2.59156584e2*TI);
    RF(78) = EXP((real)3.70803784e1 - (real)6.3e-1*ALOGT - (real)1.92731984e2*TI);
    RF(79) = (real)8.43e13;
    RF(80) = EXP((real)1.78408622e1 + (real)1.6e0*ALOGT - (real)2.72743434e3*TI);
    RF(81) = (real)2.501e13;
    RF(82) = EXP((real)3.10595094e1 - (real)1.449264e4*TI);
    RF(83) = EXP((real)2.43067848e1 - (real)4.49875701e3*TI);
    RF(84) = (real)1.e12;
    RF(85) = (real)1.34e13;
    RF(86) = EXP((real)1.01064284e1 + (real)2.47e0*ALOGT - (real)2.60666234e3*TI);
    RF(87) = (real)3.e13;
    RF(88) = (real)8.48e12;
    RF(89) = (real)1.8e13;
    RF(90) = EXP((real)8.10772006e0 + (real)2.81e0*ALOGT - (real)2.94884967e3*TI);
    RF(91) = (real)4.e13;
    TMP = EXP((real)2.86833501e2*TI);
    RF(92) = (real)1.2e13 * TMP;
    RF(107) = (real)1.6e13 * TMP;
    RF(93) = EXP((real)3.75927776e1 - (real)9.7e-1*ALOGT - (real)3.11994334e2*TI);
    RF(94) = EXP((real)2.9238457e1 + (real)1.e-1*ALOGT - (real)5.33409668e3*TI);
    RF(95) = (real)5.e13;
    RF(96) = (real)2.e13;
    RF(97) = (real)3.2e13;
    RF(98) = (real)1.6e13;
    RF(99) = (real)1.e13;
    RF(100) = (real)5.e12;
    RF(101) = EXP((real)-2.84796532e1 + (real)7.6e0*ALOGT + (real)1.77635484e3*TI);
    RF(102) = EXP((real)2.03077504e1 + (real)1.62e0*ALOGT - (real)5.45486868e3*TI);
    RF(103) = EXP((real)2.07430685e1 + (real)1.5e0*ALOGT - (real)4.32766334e3*TI);
    RF(104) = EXP((real)1.84206807e1 + (real)1.6e0*ALOGT - (real)1.570036e3*TI);
    RF(105) = (real)6.e13;
    RF(106) = EXP((real)1.47156719e1 + (real)2.e0*ALOGT - (real)4.16160184e3*TI);
    RF(108) = (real)1.e14;
    RF(109) = (real)1.e14;
    RF(110) = EXP((real)2.81010247e1 - (real)4.29747034e2*TI);
    RF(111) = (real)5.e13;
    RF(112) = (real)3.e13;
    RF(113) = (real)1.e13;
    RF(114) = EXP((real)3.43156328e1 - (real)5.2e-1*ALOGT - (real)2.55382459e4*TI);
    RF(115) = EXP((real)1.97713479e1 + (real)1.62e0*ALOGT - (real)1.86432818e4*TI);
    TMP = EXP((real)2.e0*ALOGT - (real)9.56111669e2*TI );
    RF(116) = (real)1.632e7 * TMP;
    RF(117) = (real)4.08e6 * TMP;
    RF(118) = EXP((real)-8.4310155e0 + (real)4.5e0*ALOGT + (real)5.03216668e2*TI);
    RF(119) = EXP((real)-7.6354939e0 + (real)4.e0*ALOGT + (real)1.00643334e3*TI);
    RF(120) = EXP((real)1.61180957e1 + (real)2.e0*ALOGT - (real)3.01930001e3*TI);
    RF(121) = EXP((real)1.27430637e2 - (real)1.182e1*ALOGT - (real)1.79799315e4*TI);
    RF(122) = (real)1.e14;
    RF(123) = (real)1.e14;
    RF(124) = (real)2.e13;
    RF(125) = (real)1.e13;
    RF(126) = EXP((real)3.34301138e1 -(real)6.e-2*ALOGT - (real)4.27734167e3*TI);
    RF(128) = EXP((real)2.11287309e1 +(real)1.43e0*ALOGT - (real)1.35365284e3*TI);
    RF(130) = EXP((real)2.81906369e1 -(real)6.79342501e2*TI);
    TMP = EXP((real)-1.00643334e3*TI);
    RF(131) = (real)7.5e12 * TMP;
    RF(152) = (real)1.e13 * TMP;
    RF(186) = (real)2.e13 * TMP;
    RF(132) = EXP((real)2.94360258e1 + (real)2.7e-1*ALOGT - (real)1.40900667e2*TI);
    RF(133) = (real)3.e13;
    RF(134) = (real)6.e13;
    RF(135) = (real)4.8e13;
    RF(136) = (real)4.8e13;
    RF(137) = (real)3.011e13;
    RF(138) = EXP((real)1.41081802e1 +(real)1.61e0*ALOGT +(real)1.9293327e2*TI);
    RF(139) = EXP((real)2.64270483e1 +(real)2.9e-1*ALOGT -(real)5.53538334e0*TI);
    RF(140) = EXP((real)3.83674178e1 -(real)1.39e0*ALOGT -(real)5.08248834e2*TI);
    RF(141) = (real)1.e13;
    RF(142) = EXP((real)2.32164713e1 + (real)2.99917134e2*TI);
    RF(143) = (real)9.033e13;
    RF(144) = (real)3.92e11;
    RF(145) = (real)2.5e13;
    RF(146) = EXP((real)5.56675073e1 - (real)2.83e0*ALOGT - (real)9.36888792e3*TI);
    RF(147) = EXP((real)9.64601125e1 - (real)9.147e0*ALOGT - (real)2.36008617e4*TI);
    RF(148) = (real)1.e14;
    RF(149) = (real)9.e13;
    TMP = EXP((real)-2.01286667e3*TI);
    RF(150) = (real)2.e13 * TMP;
    RF(151) = (real)2.e13 * TMP;
    RF(153) = (real)1.4e11;
    RF(154) = (real)1.8e10;
    RF(155) = EXP((real)2.97104627e1 + (real)4.4e-1*ALOGT - (real)4.46705436e4*TI);
    RF(156) = EXP((real)2.77079822e1 + (real)4.54e-1*ALOGT - (real)9.15854335e2*TI);
    RF(157) = EXP((real)1.77414365e1 + (real)1.93e0*ALOGT - (real)6.51665585e3*TI);
    RF(158) = EXP((real)1.65302053e1 + (real)1.91e0*ALOGT - (real)1.88203034e3*TI);
    TMP = EXP( (real)1.83e0*ALOGT - (real)1.10707667e2*TI );
    RF(159) = (real)1.92e7 * TMP;
    RF(160) = (real)3.84e5 * TMP;
    RF(161) = EXP((real)1.50964444e1 + (real)2.e0*ALOGT - (real)1.25804167e3*TI);
    RF(162) = EXP((real)3.13734413e1 - (real)3.05955734e4*TI);
    RF(163) = EXP((real)2.83241683e1 - (real)7.04503335e3*TI);
    RF(164) = EXP((real)1.61180957e1 + (real)2.e0*ALOGT - (real)4.02573334e3*TI);
    RF(165) = EXP((real)3.06267534e1 - (real)3.01930001e3*TI);
    RF(166) = (real)5.e13;
    RF(167) = (real)5.e13;
    RF(168) = EXP((real)1.23327053e1 + (real)2.e0*ALOGT - (real)4.62959334e3*TI);
    RF(169) = EXP((real)2.65223585e1 - (real)3.87476834e3*TI);
    RF(170) = EXP((real)4.07945264e1 - (real)9.9e-1*ALOGT - (real)7.95082335e2*TI);
    RF(171) = (real)2.e12;
    RF(172) = (real)1.604e13;
    RF(173) = (real)8.02e13;
    RF(174) = (real)2.e10;
    RF(175) = (real)3.e11;
    RF(176) = (real)3.e11;
    RF(177) = (real)2.4e13;
    RF(178) = EXP((real)2.28865889e1 - (real)4.90133034e2*TI);
    RF(179) = (real)1.2e14;
    RF(180) = EXP((real)1.85604427e1 + (real)1.9e0*ALOGT  - (real)3.78922151e3*TI);
    RF(181) = EXP((real)1.83130955e1 + (real)1.92e0*ALOGT - (real)2.86330284e3*TI);
    RF(182) = EXP((real)1.50796373e1 + (real)2.12e0*ALOGT - (real)4.37798501e2*TI);
    RF(183) = EXP((real)3.13199006e1 + (real)2.76769167e2*TI);
    RF(184) = EXP((real)1.56303353e1 + (real)1.74e0*ALOGT - (real)5.25861418e3*TI);
    RF(185) = (real)2.e14;
    RF(187) = (real)2.66e12;
    RF(188) = (real)6.6e12;
    RF(189) = (real)6.e13;
    RF(190) = EXP((real)3.02187852e1 -(real)1.64083859e3*TI);
    RF(191) = EXP((real)5.11268757e1 -(real)2.39e0*ALOGT - (real)5.62596234e3*TI);
    RF(192) = EXP((real)1.20435537e1 +(real)2.5e0*ALOGT - (real)1.2530095e3*TI);
    RF(193) = EXP((real)1.86030023e1 +(real)1.65e0*ALOGT - (real)1.6455185e2*TI);
    RF(194) = EXP((real)1.73708586e1 +(real)1.65e0*ALOGT + (real)4.89126601e2*TI);
    RF(195) = EXP((real)2.59162227e1 +(real)7.e-1*ALOGT - (real)2.95891401e3*TI);
    RF(196) = EXP((real)1.49469127e1 +(real)2.e0*ALOGT + (real)1.49958567e2*TI);
    RF(197) = EXP((real)9.16951838e0 +(real)2.6e0*ALOGT - (real)6.99974385e3*TI);
    RF(198) = EXP((real)7.8845736e-1 +(real)3.5e0*ALOGT - (real)2.85575459e3*TI);
    RF(199) = EXP((real)5.65703751e1 -(real)2.92e0*ALOGT - (real)6.29272443e3*TI);
    RF(200) = (real)1.8e12;
    RF(201) = (real)9.6e13;
    RF(202) = (real)2.4e13;
    RF(203) = (real)9.e10;
    RF(204) = (real)2.4e13;
    RF(205) = (real)1.1e13;
    RF(206) = EXP((real)7.50436995e1 -(real)5.22e0*ALOGT -(real)9.93701954e3*TI);
}

template <class t1, class t2, class t3, class t4, class t5>
__device__ t1 polyx(t1 x, t2 c0, t3 c1, t4 c2, t5 c3)
{
    return (((c3 * x + c2) * x + c1) * x + c0) * x;
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RDSMH_THRD, RDSMH_BLK)
rdsmh_kernel(const real* RESTRICT T, real* RESTRICT EG, real TCONV)
{

    const real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;
    const real TLOG = LOG((TEMP));
    const real TI = (real)1.0e0/(TEMP);

    const real TN1 = TLOG - (real)1.0;

    if ((TEMP) > (real)1.0e3)
    {
        EG(1) = EXP((real)-3.20502331e+00 + (real)9.50158922e+02*TI
                    + (real)3.33727920e+00*TN1 + polyx (TEMP,
                    (real)- 2.47012365e-05, (real)8.32427963e-08,
                    (real)- 1.49638662e-11, (real)1.00127688e-15));
        EG(2) = EXP((real)-4.46682914e-01 - (real)2.54736599e+04*TI
                    + (real)2.50000001e+00*TN1 + polyx (TEMP,
                    (real)- 1.15421486e-11, (real)2.69269913e-15,
                    (real)- 3.94596029e-19, (real)2.49098679e-23));
        EG(3) = EXP((real)4.78433864e+00 - (real)2.92175791e+04*TI
                    + (real)2.56942078e+00*TN1 + polyx (TEMP,
                    (real)- 4.29870569e-05, (real)6.99140982e-09,
                    (real)- 8.34814992e-13, (real)6.14168455e-17));
        EG(4) = EXP((real)5.45323129e+00 + (real)1.08845772e+03*TI
                    + (real)3.28253784e+00*TN1 + polyx (TEMP,
                    (real) 7.41543770e-04, (real)- 1.26327778e-07,
                    (real) 1.74558796e-11, (real)- 1.08358897e-15));
        EG(5) = EXP((real)4.47669610e+00 - (real)3.85865700e+03*TI
                    + (real)3.09288767e+00*TN1 + polyx (TEMP,
                    (real) 2.74214858e-04, (real)2.10842047e-08,
                    (real)- 7.32884630e-12, (real)5.87061880e-16));
        EG(6) = EXP((real)4.96677010e+00 + (real)3.00042971e+04*TI
                    + (real)3.03399249e+00*TN1 + polyx (TEMP,
                    + (real)1.08845902e-03, (real)-2.73454197e-08,
                    (real)- 8.08683225e-12, (real)8.41004960e-16));
        EG(7) = EXP((real)3.78510215e+00 - (real)1.11856713e+02*TI
                    + (real)4.01721090e+00*TN1 + polyx (TEMP,
                    + (real)1.11991007e-03, (real)-1.05609692e-07,
                    + (real)9.52053083e-12, (real)-5.39542675e-16));
        EG(8) = EXP((real)2.91615662e+00 + (real)1.78617877e+04*TI
                    + (real)4.16500285e+00*TN1 + polyx (TEMP,
                    + (real)2.45415847e-03, (real)-3.16898708e-07,
                    + (real)3.09321655e-11, (real)-1.43954153e-15));
        EG(9) = EXP((real)5.48497999e+00 - (real)7.10124364e+04*TI
                    + (real)2.87846473e+00*TN1 + polyx (TEMP,
                    + (real)4.85456841e-04, (real)2.40742758e-08,
                    (real)- 1.08906541e-11, (real)8.80396915e-16));
        EG(10) = EXP((real)6.17119324e+00 - (real)4.62636040e+04*TI
                     + (real)2.87410113e+00*TN1 + polyx (TEMP,
                     + (real)1.82819646e-03, (real)-2.34824328e-07,
                     + (real)2.16816291e-11, (real)-9.38637835e-16));
        EG(11) = EXP((real)8.62650169e+00 - (real)5.09259997e+04*TI
                     + (real)2.29203842e+00*TN1 + polyx (TEMP,
                     + (real)2.32794319e-03, (real)-3.35319912e-07,
                     + (real)3.48255000e-11, (real)-1.69858183e-15));
        EG(12) = EXP((real)8.48007179e+00 - (real)1.67755843e+04*TI
                     + (real)2.28571772e+00*TN1 + polyx (TEMP,
                     + (real)3.61995018e-03, (real)-4.97857247e-07,
                     + (real)4.96403870e-11, (real)-2.33577197e-15));
        EG(13) = EXP((real)1.84373180e+01 + (real)9.46834459e+03*TI
                     + (real)7.48514950e-02*TN1 + polyx (TEMP,
                     + (real)6.69547335e-03, (real)-9.55476348e-07,
                     + (real)1.01910446e-10, (real)-5.09076150e-15));
        EG(14) = EXP((real)7.81868772e+00 + (real)1.41518724e+04*TI
                     + (real)2.71518561e+00*TN1 + polyx (TEMP,
                     + (real)1.03126372e-03, (real)-1.66470962e-07,
                     + (real)1.91710840e-11, (real)-1.01823858e-15));
        EG(15) = EXP((real)2.27163806e+00 + (real)4.87591660e+04*TI
                     + (real)3.85746029e+00*TN1 + polyx (TEMP,
                     + (real)2.20718513e-03, (real)-3.69135673e-07,
                     + (real)4.36241823e-11, (real)-2.36042082e-15));
        EG(16) = EXP((real)9.79834492e+00 - (real)4.01191815e+03*TI
                     + (real)2.77217438e+00*TN1 + polyx (TEMP,
                     + (real)2.47847763e-03, (real)-4.14076022e-07,
                     + (real)4.90968148e-11, (real)-2.66754356e-15));
        EG(17) = EXP((real)1.36563230e+01 + (real)1.39958323e+04*TI
                     + (real)1.76069008e+00*TN1 + polyx (TEMP,
                     + (real)4.60000041e-03, (real)-7.37098022e-07,
                     + (real)8.38676767e-11, (real)-4.41927820e-15));
        EG(18) = EXP((real)2.92957500e+00 - (real)1.27832520e+02*TI
                     + (real)3.77079900e+00*TN1 + polyx (TEMP,
                     + (real)3.93574850e-03, (real)-4.42730667e-07,
                     + (real)3.28702583e-11, (real)-1.05630800e-15));
        EG(19) = EXP((real)-1.23028121e+00 - (real)2.59359992e+04*TI
                     + (real)4.14756964e+00*TN1 + polyx (TEMP,
                     + (real)2.98083332e-03, (real)-3.95491420e-07,
                     + (real)3.89510143e-11, (real)-1.80617607e-15));
        EG(20) = EXP((real)6.40237010e-01 - (real)4.83166880e+04*TI
                     + (real)4.27803400e+00*TN1 + polyx (TEMP,
                     + (real)2.37814020e-03, (real)-2.71683483e-07,
                     + (real)2.12190050e-11, (real)-7.44318950e-16));
        EG(21) = EXP((real)7.78732378e+00 - (real)3.46128739e+04*TI
                     + (real)3.01672400e+00*TN1 + polyx (TEMP,
                     + (real)5.16511460e-03, (real)-7.80137248e-07,
                     + (real)8.48027400e-11, (real)-4.31303520e-15));
        EG(22) = EXP((real)1.03053693e+01 - (real)4.93988614e+03*TI
                     + (real)2.03611116e+00*TN1 + polyx (TEMP,
                     + (real)7.32270755e-03, (real)-1.11846319e-06,
                     + (real)1.22685769e-10, (real)-6.28530305e-15));
        EG(23) = EXP((real)1.34624343e+01 - (real)1.28575200e+04*TI
                     + (real)1.95465642e+00*TN1 + polyx (TEMP,
                     + (real)8.69863610e-03, (real)-1.33034445e-06,
                     + (real)1.46014741e-10, (real)-7.48207880e-15));
        EG(24) = EXP((real)1.51156107e+01 + (real)1.14263932e+04*TI
                     + (real)1.07188150e+00*TN1 + polyx (TEMP,
                     + (real)1.08426339e-02, (real)-1.67093445e-06,
                     + (real)1.84510001e-10, (real)-9.50014450e-15));
        EG(25) = EXP((real)-3.93025950e+00 - (real)1.93272150e+04*TI
                     + (real)5.62820580e+00*TN1 + polyx (TEMP,
                     + (real)2.04267005e-03, (real)-2.65575783e-07,
                     + (real)2.38550433e-11, (real)-9.70391600e-16));
        EG(26) = EXP((real)6.32247205e-01 + (real)7.55105311e+03*TI
                     + (real)4.51129732e+00*TN1 + polyx (TEMP,
                     + (real)4.50179872e-03, (real)-6.94899392e-07,
                     + (real)7.69454902e-11, (real)-3.97419100e-15));
        EG(27) = EXP((real)-5.03208790e+00 - (real)4.90321780e+02*TI
                     + (real)5.97566990e+00*TN1 + polyx (TEMP,
                     + (real)4.06529570e-03, (real)-4.57270750e-07,
                     + (real)3.39192008e-11, (real)-1.08800855e-15));
        EG(28) = EXP((real)-3.48079170e+00 + (real)2.25931220e+04*TI
                     + (real)5.40411080e+00*TN1 + polyx (TEMP,
                     + (real)5.86152950e-03, (real)-7.04385617e-07,
                     + (real)5.69770425e-11, (real)-2.04924315e-15));
        EG(29) = EXP((real)-1.12430500e+01 - (real)1.74824490e+04*TI
                     + (real)6.50078770e+00*TN1 + polyx (TEMP,
                     + (real)7.16236550e-03, (real)-9.46360533e-07,
                     + (real)9.23400083e-11, (real)-4.51819435e-15));
        EG(30) = EXP((real)-1.33133500e+01 + (real)9.23570300e+02*TI
                     + (real)6.73225700e+00*TN1 + polyx (TEMP,
                     + (real)7.45417000e-03, (real)-8.24983167e-07,
                     + (real)6.01001833e-11, (real)-1.88310200e-15));
        EG(31) = EXP((real)-1.55152970e+01 - (real)7.97622360e+03*TI
                     + (real)7.70974790e+00*TN1 + polyx (TEMP,
                     + (real)8.01574250e-03, (real)-8.78670633e-07,
                     + (real)6.32402933e-11, (real)-1.94313595e-15));
    }
    else
    {
        EG(1) = EXP((real)6.83010238e-01 + (real)9.17935173e+02*TI
                    + (real)2.34433112e+00*TN1 + polyx (TEMP,
                    + (real)3.99026037e-03, (real)-3.24635850e-06,
                    + (real)1.67976745e-09, (real)-3.68805881e-13));
        EG(2) = EXP((real)-4.46682853e-01 - (real)2.54736599e+04*TI
                    + (real)2.50000000e+00*TN1 + polyx (TEMP,
                    + (real)3.52666409e-13, (real)-3.32653273e-16,
                    + (real)1.91734693e-19, (real)-4.63866166e-23));
        EG(3) = EXP((real)2.05193346e+00 - (real)2.91222592e+04*TI
                    + (real)3.16826710e+00*TN1 + polyx (TEMP,
                    (real)- 1.63965942e-03, (real)1.10717733e-06,
                    (real)- 5.10672187e-10, (real)1.05632986e-13));
        EG(4) = EXP((real)3.65767573e+00 + (real)1.06394356e+03*TI
                    + (real)3.78245636e+00*TN1 + polyx (TEMP,
                    (real)- 1.49836708e-03, (real)1.64121700e-06,
                    (real)- 8.06774591e-10, (real)1.62186419e-13));
        EG(5) = EXP((real)-1.03925458e-01 - (real)3.61508056e+03*TI
                    + (real)3.99201543e+00*TN1 + polyx (TEMP,
                    (real)- 1.20065876e-03, (real)7.69656402e-07,
                    (real)- 3.23427778e-10, (real)6.82057350e-14));
        EG(6) = EXP((real)-8.49032208e-01 + (real)3.02937267e+04*TI
                    + (real)4.19864056e+00*TN1 + polyx (TEMP,
                    (real)- 1.01821705e-03, (real)1.08673369e-06,
                    (real)- 4.57330885e-10, (real)8.85989085e-14));
        EG(7) = EXP((real)3.71666245e+00 - (real)2.94808040e+02*TI
                    + (real)4.30179801e+00*TN1 + polyx (TEMP,
                    (real)- 2.37456025e-03, (real)3.52638152e-06,
                    (real)- 2.02303245e-09, (real)4.64612562e-13));
        EG(8) = EXP((real)3.43505074e+00 + (real)1.77025821e+04*TI
                    + (real)4.27611269e+00*TN1 + polyx (TEMP,
                    (real)- 2.71411208e-04, (real)2.78892835e-06,
                    (real)- 1.79809011e-09, (real)4.31227182e-13));
        EG(9) = EXP((real)2.08401108e+00 - (real)7.07972934e+04*TI
                    + (real)3.48981665e+00*TN1 + polyx (TEMP,
                    + (real)1.61917771e-04, (real)-2.81498442e-07,
                    + (real)2.63514439e-10, (real)-7.03045335e-14));
        EG(10) = EXP((real)1.56253185e+00 - (real)4.60040401e+04*TI
                     + (real)3.76267867e+00*TN1 + polyx (TEMP,
                     + (real)4.84436072e-04, (real)4.65816402e-07,
                     (real)- 3.20909294e-10, (real)8.43708595e-14));
        EG(11) = EXP((real)-7.69118967e-01 - (real)5.04968163e+04*TI
                     + (real)4.19860411e+00*TN1 + polyx (TEMP,
                     (real)- 1.18330710e-03, (real)1.37216037e-06,
                     (real)- 5.57346651e-10, (real)9.71573685e-14));
        EG(12) = EXP((real)1.60456433e+00 - (real)1.64449988e+04*TI
                     + (real)3.67359040e+00*TN1 + polyx (TEMP,
                     + (real)1.00547588e-03, (real)9.55036427e-07,
                     (real)- 5.72597854e-10, (real)1.27192867e-13));
        EG(13) = EXP((real)-4.64130376e+00 + (real)1.02466476e+04*TI
                     + (real)5.14987613e+00*TN1 + polyx (TEMP,
                     (real)- 6.83548940e-03, (real)8.19667665e-06,
                     (real)- 4.03952522e-09, (real)8.33469780e-13));
        EG(14) = EXP((real)3.50840928e+00 + (real)1.43440860e+04*TI
                     + (real)3.57953347e+00*TN1 + polyx (TEMP,
                     (real)- 3.05176840e-04, (real)1.69469055e-07,
                     + (real)7.55838237e-11, (real)-4.52212249e-14));
        EG(15) = EXP((real)9.90105222e+00 + (real)4.83719697e+04*TI
                     + (real)2.35677352e+00*TN1 + polyx (TEMP,
                     + (real)4.49229839e-03, (real)-1.18726045e-06,
                     + (real)2.04932518e-10, (real)-7.18497740e-15));
        EG(16) = EXP((real)3.39437243e+00 - (real)3.83956496e+03*TI
                     + (real)4.22118584e+00*TN1 + polyx (TEMP,
                     (real)- 1.62196266e-03, (real)2.29665743e-06,
                     (real)- 1.10953411e-09, (real)2.16884433e-13));
        EG(17) = EXP((real)6.02812900e-01 + (real)1.43089567e+04*TI
                     + (real)4.79372315e+00*TN1 + polyx (TEMP,
                     (real)- 4.95416685e-03, (real)6.22033347e-06,
                     (real)- 3.16071051e-09, (real)6.58863260e-13));
        EG(18) = EXP((real)1.31521770e+01 - (real)9.78601100e+02*TI
                     + (real)2.10620400e+00*TN1 + polyx (TEMP,
                     + (real)3.60829750e-03, (real)8.89745333e-07,
                     (real)- 6.14803000e-10, (real)1.03780500e-13));
        EG(19) = EXP((real)1.39397051e+01 - (real)2.64289807e+04*TI
                     + (real)8.08681094e-01*TN1 + polyx (TEMP,
                     + (real)1.16807815e-02, (real)-5.91953025e-06,
                     + (real)2.33460364e-09, (real)-4.25036487e-13));
        EG(20) = EXP((real)5.92039100e+00 - (real)4.86217940e+04*TI
                     + (real)3.28154830e+00*TN1 + polyx (TEMP,
                     + (real)3.48823955e-03, (real)-3.97587400e-07,
                     (real)- 1.00870267e-10, (real)4.90947725e-14));
        EG(21) = EXP((real)8.51054025e+00 - (real)3.48598468e+04*TI
                     + (real)3.21246645e+00*TN1 + polyx (TEMP,
                     + (real)7.57395810e-04, (real)4.32015687e-06,
                     (real)- 2.98048206e-09, (real)7.35754365e-13));
        EG(22) = EXP((real)4.09733096e+00 - (real)5.08977593e+03*TI
                     + (real)3.95920148e+00*TN1 + polyx (TEMP,
                     (real)- 3.78526124e-03, (real)9.51650487e-06,
                     (real)- 5.76323961e-09, (real)1.34942187e-12));
        EG(23) = EXP((real)4.70720924e+00 - (real)1.28416265e+04*TI
                     + (real)4.30646568e+00*TN1 + polyx (TEMP,
                     (real)- 2.09329446e-03, (real)8.28571345e-06,
                     (real)- 4.99272172e-09, (real)1.15254502e-12));
        EG(24) = EXP((real)2.66682316e+00 + (real)1.15222055e+04*TI
                     + (real)4.29142492e+00*TN1 + polyx (TEMP,
                     (real)- 2.75077135e-03, (real)9.99063813e-06,
                     (real)- 5.90388571e-09, (real)1.34342886e-12));
        EG(25) = EXP((real)1.24904170e+01 - (real)2.00594490e+04*TI
                     + (real)2.25172140e+00*TN1 + polyx (TEMP,
                     + (real)8.82751050e-03, (real)-3.95485017e-06,
                     + (real)1.43964658e-09, (real)-2.53324055e-13));
        EG(26) = EXP((real)1.22156480e+01 + (real)7.04291804e+03*TI
                     + (real)2.13583630e+00*TN1 + polyx (TEMP,
                     + (real)9.05943605e-03, (real)-2.89912457e-06,
                     + (real)7.78664640e-10, (real)-1.00728807e-13));
        EG(27) = EXP((real)9.57145350e+00 - (real)1.52147660e+03*TI
                     + (real)3.40906240e+00*TN1 + polyx (TEMP,
                     + (real)5.36928700e-03, (real)3.15248750e-07,
                     + (real)5.96548592e-10, (real)1.43369255e-13));
        EG(28) = EXP((real)4.10301590e+00 + (real)2.15728780e+04*TI
                     + (real)4.72945950e+00*TN1 + polyx (TEMP,
                     (real)- 1.59664290e-03, (real)7.92248683e-06,
                     (real)- 4.78821758e-09, (real)1.09655560e-12));
        EG(29) = EXP((real)1.71732140e+01 - (real)1.92456290e+04*TI
                     + (real)1.36318350e+00*TN1 + polyx (TEMP,
                     + (real)9.90691050e-03, (real)2.08284333e-06,
                     (real)- 2.77962958e-09, (real)7.92328550e-13));
        EG(30) = EXP((real)1.61453400e+01 - (real)1.07482600e+03*TI
                     + (real)1.49330700e+00*TN1 + polyx (TEMP,
                     (real) 1.04625900e-02, (real)7.47799000e-07,
                     (real)- 1.39076000e-09, (real)3.57907300e-13));
        EG(31) = EXP((real)2.11360340e+01 - (real)1.03123460e+04*TI
                     + (real)1.04911730e+00*TN1 + polyx (TEMP,
                     (real) 1.30044865e-02, (real)3.92375267e-07,
                     (real)- 1.63292767e-09, (real)4.68601035e-13));
    }
}


#endif //RATT_H
