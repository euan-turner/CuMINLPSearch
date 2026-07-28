$offlisting
*  
*  Equation counts
*      Total        E        G        L        N        X        C        B
*          1        1        0        0        0        0        0        0
*  
*  Variable counts
*                   x        b        i      s1s      s2s       sc       si
*      Total     cont   binary  integer     sos1     sos2    scont     sint
*          3        3        0        0        0        0        0        0
*  FX      0
*  
*  Nonzero counts
*      Total    const       NL      DLL
*          3        1        2        0
*
*  Solve m using NLP minimizing objvar;


Variables  x1,x2,objvar;

Equations  e1;


e1.. -(2*sqr(x1) - 1.05*POWER(x1,4) + 0.166666666666667*POWER(x1,6) - x1*x2 + 
     sqr(x2)) + objvar =E= 0;

* set non-default bounds
x1.lo = -5;
x2.up = 5;

Model m / all /;

m.limrow=0; m.limcol=0;
m.tolproj=0.0;

$if NOT '%gams.u1%' == '' $include '%gams.u1%'

$if not set NLP $set NLP NLP
Solve m using %NLP% minimizing objvar;
