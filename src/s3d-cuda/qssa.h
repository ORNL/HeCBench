#ifndef QSSA_H
#define QSSA_H

#include "S3D.h"

template <class real>
__global__ void
LAUNCH_BOUNDS (QSSA_THRD, QSSA_BLK)
qssa_kernel(real* RESTRICT RF, real* RESTRICT RB, real* RESTRICT A) {

    real DEN;

    RF(57)  = (real)0.0;
    RF(58)  = (real)0.0;
    RF(143) = (real)0.0;
    RF(179) = (real)0.0;
    RB(194) = (real)0.0;
    RF(206) = (real)0.0;

    //   CH
    DEN = +RF(34) +RF(35) +RF(36) +RF(37) +RF(38)
               +RF(39) +RF(40) +RF(77) +RF(87) +RF(105) +RF(111)
               +RB(54) +RB(60) ;
    A(1,0) = DIV ((+RB(34) +RB(37) +RB(39) +RB(57) +RB(77)
            +RB(105) +RB(111)), DEN);
    A(1,2) = DIV ((+RB(36) +RF(54)), DEN);
    A(1,3) = DIV ((+RF(60)), DEN);
    A(1,4) = DIV ((+RB(35) +RB(38) +RB(40)), DEN);
    A(1,7) = DIV ((+RB(87)), DEN);
    //   CH2
    DEN = +RF(48) +RF(49) +RF(50) +RF(51) +RF(52)
               +RF(53) +RF(54) +RF(55) +RF(56) +RF(91) +RF(106)
               +RF(112) +RF(165) +RB(36) +RB(59) +RB(67) +RB(68)
               +RB(69) +RB(80) +RB(117) +RB(123) +RB(125) +RB(130)
               +RB(160) ;
    A(2,0) = DIV ((+RB(48) +RB(49) +RB(52) +RB(53) +RB(55)
            +RB(56) +RB(57) +RB(58) +RB(58) +RF(80) +RB(91)
            +RB(106) +RF(117) +RF(130) +RF(160) +RB(165)), DEN);
    A(2,1) = DIV ((+RF(36) +RB(54)), DEN);
    A(2,3) = DIV ((+RF(59) +RF(67) +RF(68) +RF(69)), DEN);
    A(2,4) = DIV ((+RB(50) +RB(51)), DEN);
    A(2,6) = DIV ((+RF(123) +RF(125)), DEN);
    A(2,7) = DIV ((+RB(112)), DEN);
    //   CH2*
    DEN = +RF(59) +RF(60) +RF(61) +RF(62) +RF(63)
               +RF(64) +RF(65) +RF(66) +RF(67) +RF(68) +RF(69)
               +RF(70) +RF(92) +RF(107) +RF(166) +RF(167) +RF(183)
               +RB(81) +RB(98) +RB(108) ;
    A(3,0) = DIV ((+RB(61) +RB(63) +RB(64) +RB(65) +RB(66)
            +RB(70) +RF(81) +RB(92) +RB(107) +RF(108) +RB(167)), DEN);
    A(3,1) = DIV ((+RB(60)), DEN);
    A(3,2) = DIV ((+RB(59) +RB(67) +RB(68) +RB(69)), DEN);
    A(3,4) = DIV ((+RB(62)), DEN);
    A(3,5) = DIV ((+RF(98)), DEN);
    A(3,6) = DIV ((+RB(166)), DEN);
    A(3,8) = DIV ((+RB(183)), DEN);
    //     HCO
    DEN = +RF(41) +RF(42) +RF(43) +RF(44) +RF(45)
               +RF(46) +RF(47) +RF(88) +RF(89) +RF(120) +RF(164)
               +RF(189) +RB(35) +RB(38) +RB(40) +RB(50) +RB(51)
               +RB(62) +RB(72) +RB(73) +RB(74) +RB(75) +RB(76)
               +RB(90) +RB(140) +RB(149) +RB(159) ;
    A(4,0) = DIV ((+RB(41) +RB(42) +RB(43) +RB(44) +RB(45)
            +RB(46) +RB(47) +RF(72) +RF(73) +RF(74) +RF(75)
            +RF(76) +RB(88) +RB(89) +RF(90) +RB(143) +RF(159)
            +RB(179) +RB(189) +RF(194)), DEN);
    A(4,1) = DIV ((+RF(35) +RF(38) +RF(40)), DEN);
    A(4,2) = DIV ((+RF(50) +RF(51)), DEN);
    A(4,3) = DIV ((+RF(62)), DEN);
    A(4,7) = DIV ((+RB(120) +RF(140)), DEN);
    A(4,8) = DIV ((+RB(164)), DEN);
    A(4,9) = DIV ((+RF(149)), DEN);
    //   CH3O
    DEN = +RF(96) +RF(97) +RF(98) +RF(99) +RF(100)
               +RF(101) +RB(71) +RB(82) +RB(85) ;
    A(5,0) = DIV ((+RF(71) +RF(82) +RF(85) +RB(96) +RB(97)
            +RB(99) +RB(100) +RB(101)), DEN);
    A(5,3) = DIV((+RB(98)), DEN);
    //   H2CC
    DEN = +RF(122) +RF(123) +RF(124) +RF(125) +RB(114)
               +RB(134) +RB(155) +RB(166) +RB(186) ;
    A(6,0) = DIV ((+RF(114) +RB(122) +RB(124) +RF(155) +RF(186)), DEN);
    A(6,2) = DIV ((+RB(123) +RB(125)), DEN);
    A(6,3) = DIV ((+RF(166)), DEN);
    A(6,7) = DIV ((+RF(134)), DEN);
    //     C2H3
    DEN = +RF(115) +RF(132) +RF(133) +RF(134) +RF(135)
               +RF(136) +RF(137) +RF(138) +RF(139) +RF(140) +RF(141)
               +RF(142) +RF(144) +RF(145) +RF(146) +RB(87) +RB(112)
               +RB(120) +RB(157) +RB(158) +RB(161) +RB(162) +RB(168)
               +RB(188) ;
    A(7,0) = DIV ((+RB(115) +RB(132) +RB(133) +RB(135) +RB(136)
            +RB(137) +RB(138) +RB(142) +RB(143) +RB(144) +RB(145)
            +RB(146) +RF(157) +RF(158) +RF(161) +RF(162) +RF(168)
            +RF(188) +RB(206)), DEN);
    A(7,1) = DIV ((+RF(87)), DEN);
    A(7,2) = DIV ((+RF(112)), DEN);
    A(7,4) = DIV ((+RF(120) +RB(140)), DEN);
    A(7,6) = DIV ((+RB(134)), DEN);
    A(7,9) = DIV ((+RB(139) +RB(141)), DEN);
    //     C2H5
    DEN = +RF(170) +RF(171) +RF(172) +RF(173) +RF(174)
               +RF(175) +RF(176) +RF(177) +RF(178) +RB(94) +RB(156)
               +RB(164) +RB(180) +RB(181) +RB(182) +RB(183) +RB(184)
               +RB(199) +RB(201) +RB(204) ;
    A(8,0) = DIV ((+RF(94) +RF(156) +RB(170) +RB(171) +RB(172)
            +RB(173) +RB(174) +RB(175) +RB(176) +RB(177) +RB(178)
            +RB(179) +RF(180) +RF(181) +RF(182) +RF(184) +RF(194)
            +RB(206)), DEN);
    A(8,3) = DIV ((+RF(183)), DEN);
    A(8,4) = DIV ((+RF(164)), DEN);
    A(8,10) = DIV ((+RF(199) +RF(201) +RF(204)), DEN);
    //     CH2CHO
    DEN = +RF(147) +RF(148) +RF(149) +RF(150) +RF(151)
               +RF(152) +RF(153) +RF(154) +RB(126) +RB(139) +RB(141) ;
    A(9,0) = DIV ((+RF(126) +RB(147) +RB(148) +RB(150) +RB(151)
            +RB(152) +RB(153) +RB(154)), DEN);
    A(9,4) = DIV ((+RB(149)), DEN);
    A(9,7) = DIV ((+RF(139) +RF(141)), DEN);

    DEN = +RF(199) +RF(200) +RF(201) +RF(202) +RF(203)
               +RF(204) +RF(205) +RB(169) +RB(190) ;
    A(10,0) = DIV ((+RF(169) +RF(190) +RB(200) +RB(202) +RB(203)
            +RB(205)), DEN);
    A(10,8) = DIV ((+RB(199) +RB(201) +RB(204)), DEN);
}

template <class real>
__global__ void
LAUNCH_BOUNDS (QSSAB_THRD, QSSAB_BLK)
qssab_kernel(real* RESTRICT RF, real* RESTRICT RB, real* RESTRICT A)
{

    real DEN;

    A(8,0) = A(8,0) + A(8,10)*A(10,0);
    DEN = 1 -A(8,10)*A(10,8);
    A(8,0) = DIV (A(8,0), DEN);
    A(8,4) = DIV (A(8,4), DEN);
    A(8,3) = DIV (A(8,3), DEN);
    A(3,0) = A(3,0) + A(3,5)*A(5,0);
    DEN = 1 -A(3,5)*A(5,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,2) = DIV (A(3,2), DEN);
    A(3,1) = DIV (A(3,1), DEN);
    A(3,8) = DIV (A(3,8), DEN);
    A(3,6) = DIV (A(3,6), DEN);
    A(4,0) = A(4,0) + A(4,9)*A(9,0);
    A(4,7) = A(4,7) + A(4,9)*A(9,7);
    DEN = 1 -A(4,9)*A(9,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(4,7) = DIV (A(4,7), DEN);
    A(4,2) = DIV (A(4,2), DEN);
    A(4,1) = DIV (A(4,1), DEN);
    A(4,8) = DIV (A(4,8), DEN);
    A(7,0) = A(7,0) + A(7,9)*A(9,0);
    A(7,4) = A(7,4) + A(7,9)*A(9,4);
    DEN = 1 -A(7,9)*A(9,7);
    A(7,0) = DIV (A(7,0), DEN);
    A(7,4) = DIV (A(7,4), DEN);
    A(7,2) = DIV (A(7,2), DEN);
    A(7,1) = DIV (A(7,1), DEN);
    A(7,6) = DIV (A(7,6), DEN);
    A(3,0) = A(3,0) + A(3,6)*A(6,0);
    A(3,7) = A(3,6)*A(6,7);
    A(3,2) = A(3,2) + A(3,6)*A(6,2);
    DEN = 1 -A(3,6)*A(6,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,7) = DIV (A(3,7), DEN);
    A(3,2) = DIV (A(3,2), DEN);
    A(3,1) = DIV (A(3,1), DEN);
    A(3,8) = DIV (A(3,8), DEN);
    A(7,0) = A(7,0) + A(7,6)*A(6,0);
    A(7,3) = A(7,6)*A(6,3);
    A(7,2) = A(7,2) + A(7,6)*A(6,2);
    DEN = 1 -A(7,6)*A(6,7);
    A(7,0) = DIV (A(7,0), DEN);
    A(7,4) = DIV (A(7,4), DEN);
    A(7,3) = DIV (A(7,3), DEN);
    A(7,2) = DIV (A(7,2), DEN);
    A(7,1) = DIV (A(7,1), DEN);
    A(2,0) = A(2,0) + A(2,6)*A(6,0);
    A(2,3) = A(2,3) + A(2,6)*A(6,3);
    A(2,7) = A(2,7) + A(2,6)*A(6,7);
    DEN = 1 -A(2,6)*A(6,2);
    A(2,0) = DIV (A(2,0), DEN);
    A(2,4) = DIV (A(2,4), DEN);
    A(2,3) = DIV (A(2,3), DEN);
    A(2,7) = DIV (A(2,7), DEN);
    A(2,1) = DIV (A(2,1), DEN);
    A(4,0) = A(4,0) + A(4,8)*A(8,0);
    A(4,3) = A(4,3) + A(4,8)*A(8,3);
    DEN = 1 -A(4,8)*A(8,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(4,7) = DIV (A(4,7), DEN);
    A(4,2) = DIV (A(4,2), DEN);
    A(4,1) = DIV (A(4,1), DEN);
    A(3,0) = A(3,0) + A(3,8)*A(8,0);
    A(3,4) = A(3,4) + A(3,8)*A(8,4);
    DEN = 1 -A(3,8)*A(8,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,7) = DIV (A(3,7), DEN);
    A(3,2) = DIV (A(3,2), DEN);
    A(3,1) = DIV (A(3,1), DEN);
    A(4,0) = A(4,0) + A(4,1)*A(1,0);
    A(4,3) = A(4,3) + A(4,1)*A(1,3);
    A(4,7) = A(4,7) + A(4,1)*A(1,7);
    A(4,2) = A(4,2) + A(4,1)*A(1,2);
    DEN = 1 -A(4,1)*A(1,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(4,7) = DIV (A(4,7), DEN);
    A(4,2) = DIV (A(4,2), DEN);
    A(3,0) = A(3,0) + A(3,1)*A(1,0);
    A(3,4) = A(3,4) + A(3,1)*A(1,4);
    A(3,7) = A(3,7) + A(3,1)*A(1,7);
    A(3,2) = A(3,2) + A(3,1)*A(1,2);
    DEN = 1 -A(3,1)*A(1,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,7) = DIV (A(3,7), DEN);
    A(3,2) = DIV (A(3,2), DEN);
    A(7,0) = A(7,0) + A(7,1)*A(1,0);
    A(7,4) = A(7,4) + A(7,1)*A(1,4);
    A(7,3) = A(7,3) + A(7,1)*A(1,3);
    A(7,2) = A(7,2) + A(7,1)*A(1,2);
    DEN = 1 -A(7,1)*A(1,7);
    A(7,0) = DIV (A(7,0), DEN);
    A(7,4) = DIV (A(7,4), DEN);
    A(7,3) = DIV (A(7,3), DEN);
    A(7,2) = DIV (A(7,2), DEN);
    A(2,0) = A(2,0) + A(2,1)*A(1,0);
    A(2,4) = A(2,4) + A(2,1)*A(1,4);
    A(2,3) = A(2,3) + A(2,1)*A(1,3);
    A(2,7) = A(2,7) + A(2,1)*A(1,7);
    DEN = 1 -A(2,1)*A(1,2);
    A(2,0) = DIV (A(2,0), DEN);
    A(2,4) = DIV (A(2,4), DEN);
    A(2,3) = DIV (A(2,3), DEN);
    A(2,7) = DIV (A(2,7), DEN);
    A(4,0) = A(4,0) + A(4,2)*A(2,0);
    A(4,3) = A(4,3) + A(4,2)*A(2,3);
    A(4,7) = A(4,7) + A(4,2)*A(2,7);
    DEN = 1 -A(4,2)*A(2,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(4,7) = DIV (A(4,7), DEN);
    A(3,0) = A(3,0) + A(3,2)*A(2,0);
    A(3,4) = A(3,4) + A(3,2)*A(2,4);
    A(3,7) = A(3,7) + A(3,2)*A(2,7);
    DEN = 1 -A(3,2)*A(2,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,7) = DIV (A(3,7), DEN);
    A(7,0) = A(7,0) + A(7,2)*A(2,0);
    A(7,4) = A(7,4) + A(7,2)*A(2,4);
    A(7,3) = A(7,3) + A(7,2)*A(2,3);
    DEN = 1 -A(7,2)*A(2,7);
    A(7,0) = DIV (A(7,0), DEN);
    A(7,4) = DIV (A(7,4), DEN);
    A(7,3) = DIV (A(7,3), DEN);
    A(4,0) = A(4,0) + A(4,7)*A(7,0);
    A(4,3) = A(4,3) + A(4,7)*A(7,3);
    DEN = 1 -A(4,7)*A(7,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(3,0) = A(3,0) + A(3,7)*A(7,0);
    A(3,4) = A(3,4) + A(3,7)*A(7,4);
    DEN = 1 -A(3,7)*A(7,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(4,0) = A(4,0) + A(4,3)*A(3,0);
    DEN = 1 -A(4,3)*A(3,4);
    A(4,0) = DIV (A(4,0), DEN);
}


#endif //QSSA_H
