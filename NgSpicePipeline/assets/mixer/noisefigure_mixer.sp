  // Library name: Augustsecondweek2022
 // Cell name: mixer_active
// View name: schematic
 .include {model_path}
 .option TEMP=27C

mn4 Von VLOP net2 0 NMOS w=10.8u l=45.0n 
mn5 Von VLON net3 0 NMOS w=10.8u l=45.0n 
mn1 Vop VLOP net3 0 NMOS w=10.8u l=45.0n 
mn7 net2 VRFN 0 0 NMOS w=21.6u l=45.0n 
mn6 net3 VRFP 0 0 NMOS w=21.6u l=45.0n 
mn2 Vop VLON net2 0 NMOS w=10.8u l=45.0n

R0 vdd Vop   350 m=1
R1 vdd Von  350 m=1

R5 net8 net6  10 
R4 net9 VRFN  5 
R3 VRFP net10  5 

E0 (VRFP net6 net7 0) 0.5
E6 (net6 VRFN net7 0) 0.5
E5 (net5 0 Vop Von) 1.0

V1 (vdd 0) dc 1.2
V6 (net6 0) dc 725m

V2 VLOP 0 dc 0 sin(0.6 1.2 5.755G 0)
V3 VLON 0 dc 0 sin(0.6 1.2 5.755G)

V4 net7 net4 dc 0  ac 1 sin(0 1.35m 5.775G) portnum 1 z0 50
V5 net5 0 dc 0  ac 1 portnum 2 z0 50

L7 Vop Von  800p IC=700uA
L0 net7 net8  500p IC=700uA
L2 net10 Vb  500p  IC=700uA
L4 net4 net6  500p IC=700uA
L5 net9 Vb  500p  IC=700uA
L6 net5 net1  800p IC=700uA
K76  L7 L6 1
K02  L0 L2 1
K45  L4 L5 1

R77 (Vop Von)   1G
R00 (net7 net6)   1G
R22  (VRFP Vb)   1G 
R44  (net4 net6)   1G 
R55  (VRFN Vb)   1G 
R66  (net5 net1)  1G 

.control
let Vbrf_start = {Vbrf_start}
let Vbrf_stop = {Vbrf_stop}
let Vbrf_delta = {Vbrf_change}
let Vbrf_test = Vbrf_start

let RL_start = {RL_start}
let RL_stop = {RL_stop}
let RL_delta = {RL_change}
let RL_test = RL_start

let WN_start = {WN_start}
let WN_stop = {WN_stop}
let WN_delta = {WN_change}
let WN_test = WN_start

let WT_start = {WT_start}
let WT_stop = {WT_stop}
let WT_delta = {WT_change}
let WT_test = WT_start

while Vbrf_test <= Vbrf_stop
    alter V0 = Vbrf_test
    while RL_test <= RL_stop
        alter @R0[r] = RL_test
        alter @R1[r] = RL_test
        while WN_test <= WN_stop
            alter @mn1[w]=WN_test
            alter @mn5[w]=WN_test
            alter @mn4[w]=WN_test
            alter @mn2[w]=WN_test
            while WT_test <= WT_stop
                alter @mn7[w]=WT_test
			    alter @mn6[w]=WT_test
			    sp lin 1000 10e6 6e9 1 1
			    let noise_f = real(minimum(NF))
			    print noise_f >> {out}mixer_noise_figure.csv
			    set appendwriter
			    let WT_test=WT_test + WT_delta
            end
            let WT_test = WT_start
            let WN_test=WN_test + WN_delta
        end
        let WN_test = WN_start
        let RL_test=RL_test + RL_delta
    end
    let RL_test = RL_start
    let Vbrf_test=Vbrf_test + Vbrf_delta
 end
.endc