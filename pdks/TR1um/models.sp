* TR-1um device model declarations for ALIGN.
* Custom names (nmos5v/pmos5v) avoid colliding with ALIGN's builtin NMOS/PMOS.
* Base models NMOS/PMOS/RES provide pin orders; W/L/NF/M are declared here.
* W/L in meters; NF = number of fingers; M = multiplier.
.model nmos5v nmos l=1 w=1 m=1 nf=1 stack=1 parallel=1
.model pmos5v pmos l=1 w=1 m=1 nf=1 stack=1 parallel=1
.model rr res r=1
.model rs res r=1
