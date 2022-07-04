nmos: cs amp

.include C:\Spice64\lib\45nm_CS.pm

.option TEMP=27C

V1 in 0 dc 800m ac 1
Vdd net3 0 dc 1.2
R0 net3 out {1}
V2 net2 0 dc 1.2
mn0 net1 in 0 0 NMOS W={2} L=45n
mn1 out net2 net1 0 NMOS W={3} L=45n

.control

op
let gm = @mn0[gm]
let a0 = gm*r_test
let id = @mn0[id]
let pw = id*1.2
let cgd = @mn1[cgd]
let cdb = abs(@mn1[cdb])
let cp = (1+a0)*cgd+cdb
let bw = 1/(2*pi*r_test*cp)
print bw cgd cdb
wrdata bw.csv bw
wrdata pw.csv pw
wrdata a0.csv a0
set appendwrite
		
.endc

.end
