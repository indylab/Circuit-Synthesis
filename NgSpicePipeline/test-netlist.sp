nmos: cs amp

.include C:\Users\tjtom\OneDrive\Desktop\File_Cabinet\Code\SimulationStuff\45nm_CS.pm

.option TEMP=27C

V6 net2 0 dc 822.6m ac 1
Vdd net1 0 dc 1.2
R0 net1 out 620.0
C0 out 0 300f
mn0 out net2 0 0 NMOS W=2.88u L=45n

.control
op	
let gm = @mn0[gm]
let ro = 1/@mn0[gds]
let a0 = gm*R0
let id = @mn0[id]
let pw = id*1.2
let cgd = @mn0[cgd]
let cdb = abs(@mn0[cdb])
let cp = (1+a0)*cgd+cdb
let bw = 1/(2*pi*R0*cp)
wrdata bw.csv bw
wrdata pw.csv pw
wrdata a0.csv a0
set appendwrite

.endc

.end
