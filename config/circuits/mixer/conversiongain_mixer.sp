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
R6 (net7 0)  r 50 m=1

R0 (vdd Vop)  r 350 m=1
R1 (vdd Von)  r 350 m=1

E0 (VRFP net6 net7 0) 0.5
E6 (net6 VRFN net7 0) 0.5
E5 (net5 0 Vop Von) 1.0


V1 (vdd 0) dc 1.2
V0 (net6 0) dc 725m

V2 VLOP 0 dc 0 pulse (0 1.2 0 0 0 1.7376e-10 8.688e-11)
V3 VLON 0 dc 0 pulse (1.2 0 0 0 0 1.7376e-10 8.688e-11)


V4 net7 0 dc 0 ac 1 sin(0 1.35m 5.775G) portnum 1 z0 50
V5 net5 0 dc 0  ac 0 portnum 2 z0 50


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
    alter V0[dc] = Vbrf_test
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
                tran 100ns 10u
                meas tran IFpp PP v(net5) from=4e-6s to=5.5e-6s
                meas tran RFpp PP v(net7) from=4e-6s to=5.5e-6s
                print IFpp
                print RFpp
                let cgain = (IFpp / RFpp)
                print cgain >> {out}conversiongain1.csv
                set appendwriter
                let WT_test=WT_test + WT_delta

                print Vbrf_test >> {out}Vbrf.csv
                print RL_test >> {out}RL.csv
                print WN_test >> {out}WN.csv
                print WT_test >> {out}WT.csv
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

                        
