#ifndef RATT2_H
#define RATT2_H

#include "S3D.h"

//Contains kernels for the second part of the ratt routine
//These kernels can safely be executed in parallel.
template <class real>
__global__ void
LAUNCH_BOUNDS (RATT2_THRD, RATT2_BLK)
ratt2_kernel(const real* RESTRICT T, const real* RESTRICT RF, real* RESTRICT RB,
        const real* RESTRICT EG, real TCONV)
{

    real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;
    //real ALOGT = LOG(TEMP);

    const real SMALL_INV = (real)1e37;
    const real RU = (real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));

    real rtemp_inv;

    rtemp_inv = DIV ((EG(2)*EG(4)), (EG(3)*EG(5)));
    RB(1) = RF(1) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(3)), (EG(2)*EG(5)));
    RB(2) = RF(2) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(5)), (EG(2)*EG(6)));
    RB(3) = RF(3) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(5)), (EG(3)*EG(6)));
    RB(4) = RF(4) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(5) = RF(5) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(6) = RF(6) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(7) = RF(7) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(8) = RF(8) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(5)*PFAC), EG(6));
    RB(9) = RF(9) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(3)*PFAC), EG(5));
    RB(10) = RF(10) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(3)*PFAC), EG(4));
    RB(11) = RF(11) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(12) = RF(12) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(13) = RF(13) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(14) = RF(14) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(15) = RF(15) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(5)*PFAC), EG(8));
    RB(16) = RF(16) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(7)), (EG(3)*EG(6)));
    RB(17) = RF(17) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(7)), (EG(1)*EG(4)));
    RB(18) = RF(18) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(7)), (EG(5)*EG(5)));
    RB(19) = RF(19) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(7)), (EG(4)*EG(5)));
    RB(20) = RF(20) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(7)), (EG(4)*EG(6)));
    RB(21) = RF(21) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(7)), (EG(4)*EG(8)));
    RB(22) = RF(22) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(7)), (EG(4)*EG(8)));
    RB(23) = RF(23) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(8)), (EG(1)*EG(7)));
    RB(24) = RF(24) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(8)), (EG(5)*EG(6)));
    RB(25) = RF(25) * MIN(rtemp_inv, SMALL_INV);
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RATT3_THRD, RATT3_BLK)
ratt3_kernel(const real* RESTRICT T, const real* RESTRICT RF, real* RESTRICT RB,
        const real* RESTRICT EG, real TCONV)
{

    real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;
    //real ALOGT = LOG(TEMP);
    const real SMALL_INV = (real)1e37;
    const real RU=(real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));

    real rtemp_inv;

    rtemp_inv = DIV ((EG(3)*EG(8)), (EG(5)*EG(7)));
    RB(26) = RF(26) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(8)), (EG(6)*EG(7)));
    RB(27) = RF(27) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(8)), (EG(6)*EG(7)));
    RB(28) = RF(28) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(14)*PFAC), EG(15));
    RB(29) = RF(29) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(14)), (EG(2)*EG(15)));
    RB(30) = RF(30) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(14)*PFAC), EG(17));
    RB(31) = RF(31) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(14)), (EG(3)*EG(15)));
    RB(32) = RF(32) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(14)), (EG(5)*EG(15)));
    RB(33) = RF(33) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(9)), (EG(2)*EG(14)));
    RB(34) = RF(34) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(9)), (EG(2)*EG(16)));
    RB(35) = RF(35) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(9)), (EG(2)*EG(10)));
    RB(36) = RF(36) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(6)*EG(9)), (EG(2)*EG(17)));
    RB(37) = RF(37) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(9)), (EG(3)*EG(16)));
    RB(38) = RF(38) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(14)*PFAC), EG(25));
    RB(39) = RF(39) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(15)), (EG(14)*EG(16)));
    RB(40) = RF(40) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(16)*PFAC), EG(17));
    RB(41) = RF(41) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(16)), (EG(1)*EG(14)));
    RB(42) = RF(42) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(16)), (EG(5)*EG(14)));
    RB(43) = RF(43) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(16)), (EG(2)*EG(15)));
    RB(44) = RF(44) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(16)), (EG(6)*EG(14)));
    RB(45) = RF(45) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*PFAC), (EG(2)*EG(14)));
    RB(46) = RF(46) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(16)), (EG(7)*EG(14)));
    RB(47) = RF(47) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(10)*PFAC), EG(12));
    RB(48) = RF(48) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(10)), (EG(2)*EG(12)));
    RB(49) = RF(49) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(10)), (EG(2)*EG(16)));
    RB(50) = RF(50) * MIN(rtemp_inv, SMALL_INV);
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RATT4_THRD, RATT4_BLK)
ratt4_kernel(const real* RESTRICT T, const real* RESTRICT RF, real* RESTRICT RB,
        const real* RESTRICT EG, real TCONV)
{

    real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;
    //real ALOGT = LOG(TEMP);
    const real SMALL_INV = (real)1e37;
    const real RU=(real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));
    real rtemp_inv;

    rtemp_inv = DIV ((EG(4)*EG(10)), (EG(5)*EG(16)));
    RB(51) = RF(51) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(10)), (EG(2)*EG(2)*EG(15)*PFAC));
    RB(52) = RF(52) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(10)), (EG(2)*EG(17)));
    RB(53) = RF(53) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(10)), (EG(6)*EG(9)));
    RB(54) = RF(54) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(10)), (EG(5)*EG(17)));
    RB(55) = RF(55) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(14)*PFAC), EG(26));
    RB(56) = RF(56) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(10)), (EG(2)*EG(19)));
    RB(57) = RF(57) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(10)), (EG(1)*EG(19)));
    RB(58) = RF(58) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(59) = RF(59) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(11)), (EG(1)*EG(9)));
    RB(60) = RF(60) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(11)), (EG(1)*EG(14)));
    RB(61) = RF(61) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(11)), (EG(2)*EG(16)));
    RB(62) = RF(62) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(11)), (EG(2)*EG(17)));
    RB(63) = RF(63) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(11)), (EG(2)*EG(12)));
    RB(64) = RF(64) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(11)), (EG(2)*EG(5)*EG(14)*PFAC));
    RB(65) = RF(65) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(11)), (EG(6)*EG(14)));
    RB(66) = RF(66) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(67) = RF(67) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(68) = RF(68) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(69) = RF(69) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(15)), (EG(14)*EG(17)));
    RB(70) = RF(70) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(17)*PFAC), EG(18));
    RB(71) = RF(71) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(17)), (EG(1)*EG(16)));
    RB(72) = RF(72) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(17)), (EG(5)*EG(16)));
    RB(73) = RF(73) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(17)), (EG(6)*EG(16)));
    RB(74) = RF(74) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(17)), (EG(7)*EG(16)));
    RB(75) = RF(75) * MIN(rtemp_inv, SMALL_INV);
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RATT5_THRD, RATT5_BLK)
ratt5_kernel(const real* RESTRICT T, const real* RESTRICT RF, real* RESTRICT RB,
        const real* RESTRICT EG, real TCONV)
{

    const real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;
    //const real ALOGT = LOG(TEMP);
    const real SMALL_INV = (real)1e37;
    const real RU=(real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));

    real rtemp_inv;

    rtemp_inv = DIV ((EG(7)*EG(17)), (EG(8)*EG(16)));
    RB(76) = RF(76) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(17)), (EG(2)*EG(26)));
    RB(77) = RF(77) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(12)*PFAC), EG(13));
    RB(78) = RF(78) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(12)), (EG(2)*EG(17)));
    RB(79) = RF(79) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(12)), (EG(6)*EG(10)));
    RB(80) = RF(80) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(12)), (EG(6)*EG(11)));
    RB(81) = RF(81) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(12)), (EG(3)*EG(18)));
    RB(82) = RF(82) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(12)), (EG(5)*EG(17)));
    RB(83) = RF(83) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(12)), (EG(4)*EG(13)));
    RB(84) = RF(84) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(12)), (EG(5)*EG(18)));
    RB(85) = RF(85) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(8)*EG(12)), (EG(7)*EG(13)));
    RB(86) = RF(86) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(12)), (EG(2)*EG(21)));
    RB(87) = RF(87) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(16)), (EG(13)*EG(14)));
    RB(88) = RF(88) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(16)*PFAC), EG(28));
    RB(89) = RF(89) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(17)), (EG(13)*EG(16)));
    RB(90) = RF(90) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(12)), (EG(2)*EG(22)));
    RB(91) = RF(91) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(12)), (EG(2)*EG(22)));
    RB(92) = RF(92) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(12)*PFAC), EG(24));
    RB(93) = RF(93) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(12)), (EG(2)*EG(23)));
    RB(94) = RF(94) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(25)), (EG(14)*EG(22)));
    RB(95) = RF(95) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(18)), (EG(1)*EG(17)));
    RB(96) = RF(96) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(18)), (EG(5)*EG(12)));
    RB(97) = RF(97) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(18)), (EG(6)*EG(11)));
    RB(98) = RF(98) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(18)), (EG(5)*EG(17)));
    RB(99) = RF(99) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(18)), (EG(6)*EG(17)));
    RB(100) = RF(100) * MIN(rtemp_inv, SMALL_INV);

}

template <class real>
__global__ void
LAUNCH_BOUNDS (RATT6_THRD, RATT6_BLK)
ratt6_kernel(const real* RESTRICT T, const real* RESTRICT RF, real* RESTRICT RB,
        const real* RESTRICT EG, real TCONV)
{

    const real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;
    //const real ALOGT = LOG(TEMP);
    const real SMALL_INV = (real)1e37;
    const real RU = (real)8.31451e7;
    const real PATM =  (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));
    real rtemp_inv;

    rtemp_inv = DIV ((EG(4)*EG(18)), (EG(7)*EG(17)));
    RB(101) = RF(101) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(13)), (EG(1)*EG(12)));
    RB(102) = RF(102) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(13)), (EG(5)*EG(12)));
    RB(103) = RF(103) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(13)), (EG(6)*EG(12)));
    RB(104) = RF(104) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(13)), (EG(2)*EG(22)));
    RB(105) = RF(105) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(13)), (EG(12)*EG(12)));
    RB(106) = RF(106) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(13)), (EG(12)*EG(12)));
    RB(107) = RF(107) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(25)), (EG(11)*EG(14)));
    RB(108) = RF(108) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(25)), (EG(2)*EG(14)*EG(14)*PFAC));
    RB(109) = RF(109) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(25)), (EG(5)*EG(14)*EG(14)*PFAC));
    RB(110) = RF(110) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(25)), (EG(14)*EG(19)));
    RB(111) = RF(111) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(25)), (EG(14)*EG(21)));
    RB(112) = RF(112) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(25)*EG(25)), (EG(14)*EG(14)*EG(19)*PFAC));
    RB(113) = RF(113) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(19), EG(20));
    RB(114) = RF(114) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(21), (EG(2)*EG(19)*PFAC));
    RB(115) = RF(115) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(19)), (EG(2)*EG(25)));
    RB(116) = RF(116) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(19)), (EG(10)*EG(14)));
    RB(117) = RF(117) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(19)), (EG(2)*EG(26)));
    RB(118) = RF(118) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(19)), (EG(12)*EG(14)));
    RB(119) = RF(119) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(19)), (EG(14)*EG(21)));
    RB(120) = RF(120) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(19)*PFAC), EG(29));
    RB(121) = RF(121) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(19), EG(20));
    RB(122) = RF(122) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(20)), (EG(10)*EG(14)));
    RB(123) = RF(123) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(20)), (EG(2)*EG(26)));
    RB(124) = RF(124) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(20)), (EG(10)*EG(15)));
    RB(125) = RF(125) * MIN(rtemp_inv, SMALL_INV);
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RATT7_THRD, RATT7_BLK)
ratt7_kernel(const real* RESTRICT T, const real* RESTRICT RF, real* RESTRICT RB,
        const real* RESTRICT EG, real TCONV)
{
    const real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;
    //const real ALOGT = LOG(TEMP);
    const real SMALL_INV = (real)1e37;
    const real RU= (real)8.31451e7;
    const real PATM =  (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));
    real rtemp_inv;

    rtemp_inv = DIV ((EG(2)*EG(26)*PFAC), EG(27));
    RB(126) = RF(126) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(26)), (EG(1)*EG(25)));
    RB(127) = RF(127) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(26)), (EG(12)*EG(14)));
    RB(128) = RF(128) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(26)), (EG(5)*EG(25)));
    RB(129) = RF(129) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(26)), (EG(10)*EG(15)));
    RB(130) = RF(130) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(26)), (EG(6)*EG(25)));
    RB(131) = RF(131) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(21)*PFAC), EG(22));
    RB(132) = RF(132) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(21)), (EG(1)*EG(19)));
    RB(133) = RF(133) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(21)), (EG(1)*EG(20)));
    RB(134) = RF(134) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(21)), (EG(2)*EG(26)));
    RB(135) = RF(135) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(21)), (EG(12)*EG(14)));
    RB(136) = RF(136) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(21)), (EG(6)*EG(19)));
    RB(137) = RF(137) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(21)), (EG(7)*EG(19)));
    RB(138) = RF(138) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(21)), (EG(3)*EG(27)));
    RB(139) = RF(139) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(21)), (EG(16)*EG(17)));
    RB(140) = RF(140) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(21)), (EG(5)*EG(27)));
    RB(141) = RF(141) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(8)*EG(21)), (EG(7)*EG(22)));
    RB(142) = RF(142) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(21)), (EG(14)*EG(22)));
    RB(143) = RF(143) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(21)), (EG(13)*EG(19)));
    RB(144) = RF(144) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(21)*PFAC), EG(30));
    RB(145) = RF(145) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(21)), (EG(2)*EG(29)));
    RB(146) = RF(146) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(27), (EG(12)*EG(14)*PFAC));
    RB(147) = RF(147) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(27)*PFAC), EG(28));
    RB(148) = RF(148) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(27)), (EG(12)*EG(16)));
    RB(149) = RF(149) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(27)), (EG(1)*EG(26)));
    RB(150) = RF(150) * MIN(rtemp_inv, SMALL_INV);
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RATT8_THRD, RATT8_BLK)
ratt8_kernel(const real* RESTRICT T, const real* RESTRICT RF,
        real* RESTRICT RB, const real* RESTRICT EG, real TCONV)
{

    const real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;
    //const real ALOGT = LOG(TEMP);
    const real SMALL_INV = (real)1e37;
    const real RU = (real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));
    real rtemp_inv;

    rtemp_inv = DIV ((EG(3)*EG(27)), (EG(5)*EG(26)));
    RB(151) = RF(151) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(27)), (EG(6)*EG(26)));
    RB(152) = RF(152) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(27)), (EG(7)*EG(26)));
    RB(153) = RF(153) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(27)), (EG(5)*EG(14)*EG(17)*PFAC));
    RB(154) = RF(154) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(22), (EG(1)*EG(20)*PFAC));
    RB(155) = RF(155) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(22)*PFAC), EG(23));
    RB(156) = RF(156) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(22)), (EG(1)*EG(21)));
    RB(157) = RF(157) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(22)), (EG(5)*EG(21)));
    RB(158) = RF(158) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(22)), (EG(12)*EG(16)));
    RB(159) = RF(159) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(22)), (EG(10)*EG(17)));
    RB(160) = RF(160) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(22)), (EG(6)*EG(21)));
    RB(161) = RF(161) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(22)), (EG(7)*EG(21)));
    RB(162) = RF(162) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(22)), (EG(5)*EG(28)));
    RB(163) = RF(163) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(22)), (EG(14)*EG(23)));
    RB(164) = RF(164) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(22)), (EG(2)*EG(29)));
    RB(165) = RF(165) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(22)), (EG(13)*EG(20)));
    RB(166) = RF(166) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(22)), (EG(2)*EG(29)));
    RB(167) = RF(167) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(22)), (EG(13)*EG(21)));
    RB(168) = RF(168) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(22)*PFAC), EG(31));
    RB(169) = RF(169) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv =  DIV ((EG(2)*EG(23)*PFAC), EG(24));
    RB(170) = RF(170) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(23)), (EG(1)*EG(22)));
    RB(171) = RF(171) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(23)), (EG(12)*EG(17)));
    RB(172) = RF(172) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(23)), (EG(2)*EG(28)));
    RB(173) = RF(173) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(23)), (EG(7)*EG(22)));
    RB(174) = RF(174) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(23)), (EG(4)*EG(24)));
    RB(175) = RF(175) * MIN(rtemp_inv, SMALL_INV);
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RATT9_THRD, RATT9_BLK)
ratt9_kernel(const real* RESTRICT T, const real* RESTRICT RF,
        real* RESTRICT RB, const real* RESTRICT EG, real TCONV)
{

    const real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;
    //const real ALOGT = LOG(TEMP);
    const real SMALL_INV = (real)1e37;
    const real RU = (real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));
    real rtemp_inv;

    rtemp_inv = DIV ((EG(7)*EG(23)), (EG(8)*EG(22)));
    RB(176) = RF(176) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(23)), (EG(5)*EG(12)*EG(17)*PFAC));
    RB(177) = RF(177) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(8)*EG(23)), (EG(7)*EG(24)));
    RB(178) = RF(178) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(23)), (EG(14)*EG(24)));
    RB(179) = RF(179) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(24)),  (EG(1)*EG(23)));
    RB(180) = RF(180) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(24)), (EG(5)*EG(23)));
    RB(181) = RF(181) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(24)), (EG(6)*EG(23)));
    RB(182) = RF(182) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(24)), (EG(12)*EG(23)));
    RB(183) = RF(183) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(24)), (EG(13)*EG(23)));
    RB(184) = RF(184) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(29)*PFAC), EG(30));
    RB(185) = RF(185) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(29)), (EG(13)*EG(20)));
    RB(186) = RF(186) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(29)), (EG(4)*EG(30)));
    RB(187) = RF(187) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(29)), (EG(5)*EG(17)*EG(21)*PFAC));
    RB(188) = RF(188) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(29)), (EG(14)*EG(30)));
    RB(189) = RF(189) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(30)*PFAC), EG(31));
    RB(190) = RF(190) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(30)), (EG(12)*EG(22)));
    RB(191) = RF(191) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(30)), (EG(1)*EG(29)));
    RB(192) = RF(192) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(30)), (EG(2)*EG(12)*EG(26)*PFAC));
    RB(193) = RF(193) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(30)), (EG(16)*EG(23)));
    RB(194) = RF(194) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(30)), (EG(5)*EG(29)));
    RB(195) = RF(195) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(30)), (EG(6)*EG(29)));
    RB(196) = RF(196) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(30)), (EG(8)*EG(29)));
    RB(197) = RF(197) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(30)), (EG(13)*EG(29)));
    RB(198) = RF(198) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(31)), (EG(12)*EG(23)));
    RB(199) = RF(199) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(31)), (EG(1)*EG(30)));
    RB(200) = RF(200) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(31)), (EG(17)*EG(23)));
    RB(201) = RF(201) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(31)), (EG(6)*EG(30)));
    RB(202) = RF(202) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(31)), (EG(7)*EG(30)));
    RB(203) = RF(203) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(31)), (EG(5)*EG(17)*EG(23)*PFAC));
    RB(204) = RF(204) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(31)), (EG(13)*EG(30)));
    RB(205) = RF(205) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(21)*EG(23)), (EG(12)*EG(29)));
    RB(206) = RF(206) * MIN(rtemp_inv, SMALL_INV);
}

template <class real>
__global__ void
LAUNCH_BOUNDS (RATT10_THRD, RATT10_BLK)
ratt10_kernel(const real* RESTRICT T, real* RESTRICT RKLOW, real TCONV)
{

    const real TEMP = T[threadIdx.x + (blockIdx.x * blockDim.x)]*TCONV;
    const real ALOGT = LOG(TEMP);

    RKLOW(1) = EXP((real)4.22794408e1  -(real)9.e-1*ALOGT   + DIV((real)8.55468335e2,TEMP));
    RKLOW(2) = EXP((real)6.37931383e1  -(real)3.42e0*ALOGT  - DIV((real)4.24463259e4,TEMP));
    RKLOW(3) = EXP((real)6.54619238e1  -(real)3.74e0*ALOGT  - DIV((real)9.74227469e2,TEMP));
    RKLOW(4) = EXP((real)5.55621468e1  -(real)2.57e0*ALOGT  - DIV((real)7.17083751e2,TEMP));
    RKLOW(5) = EXP((real)6.33329483e1  -(real)3.14e0*ALOGT  - DIV((real)6.18956501e2,TEMP));
    RKLOW(6) = EXP((real)7.69748493e1  -(real)5.11e0*ALOGT  - DIV((real)3.57032226e3,TEMP));
    RKLOW(7) = EXP((real)6.98660102e1  -(real)4.8e0*ALOGT   - DIV((real)2.79788467e3,TEMP));
    RKLOW(8) = EXP((real)7.68923562e1  -(real)4.76e0*ALOGT  - DIV((real)1.22784867e3,TEMP));
    RKLOW(9) = EXP((real)1.11312542e2  -(real)9.588e0*ALOGT - DIV((real)2.566405e3,TEMP));
    RKLOW(10) = EXP((real)1.15700234e2 -(real)9.67e0*ALOGT  - DIV((real)3.13000767e3,TEMP));
    RKLOW(11) = EXP((real)3.54348644e1 -(real)6.4e-1*ALOGT  - DIV((real)2.50098684e4,TEMP));
    RKLOW(12) = EXP((real)6.3111756e1  -(real)3.4e0*ALOGT   - DIV((real)1.80145126e4,TEMP));
    RKLOW(13) = EXP((real)9.57409899e1 -(real)7.64e0*ALOGT  - DIV((real)5.98827834e3,TEMP));
    RKLOW(14) = EXP((real)6.9414025e1  -(real)3.86e0*ALOGT  - DIV((real)1.67067934e3,TEMP));
    RKLOW(15) = EXP((real)1.35001549e2 -(real)1.194e1*ALOGT - DIV((real)4.9163262e3,TEMP));
    RKLOW(16) = EXP((real)9.14494773e1 -(real)7.297e0*ALOGT - DIV((real)2.36511834e3,TEMP));
    RKLOW(17) = EXP((real)1.17075165e2 -(real)9.31e0*ALOGT  - DIV((real)5.02512164e4,TEMP));
    RKLOW(18) = EXP((real)9.68908955e1 -(real)7.62e0*ALOGT  - DIV((real)3.50742017e3,TEMP));
    RKLOW(19) = EXP((real)9.50941235e1 -(real)7.08e0*ALOGT  - DIV((real)3.36400342e3,TEMP));
    RKLOW(20) = EXP((real)1.38440285e2 -(real)1.2e1*ALOGT   - DIV((real)3.00309643e3,TEMP));
    RKLOW(21) = EXP((real)8.93324137e1 -(real)6.66e0*ALOGT  - DIV((real)3.52251667e3,TEMP));
}


#endif // RATT2_H
