nmos: cs amp

.include "C:\Users\tjtom\OneDrive\Desktop\File_Cabinet\Code\SimulationStuff\45nm_CS.pm"

.option TEMP=27C

V6 net2 0 dc 822.6m ac 1
Vdd net1 0 dc 1.2
R0 net1 out 500
C0 out 0 300f
mn0 out net2 0 0 NMOS W=5u L=45n

.control

let w_start = 1u
let w_stop = 5u
let delta_w = 1u
let w_test = w_start

while w_test le w_stop
	alter @mn0[w] = w_test
	op
	let gm = @mn0[gm]
	let ro = 1/@mn0[gds]
	let a0 = -gm*ro
	//print a0 w_test
	let w_test = w_test + delta_w
end

.endc

.end
