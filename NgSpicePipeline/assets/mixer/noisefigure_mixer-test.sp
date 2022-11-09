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
set Vbrf_array = ( {Vbrf_array} )
set RL_array = ( {RL_array} )
set WN_array = ( {WN_array} )
set WT_array = ( {WT_array} )

set i = {num_samples}
let index = 1
repeat $i

    alter V0 = $Vbrf_array[$&index]
    alter @R0[r] =$RL_array[$&index]
    alter @R1[r] = $RL_array[$&index]
    alter @mn1[w]=$WN_array[$&index]
    alter @mn5[w]=$WN_array[$&index]
    alter @mn4[w]=$WN_array[$&index]
    alter @mn2[w]=$WN_array[$&index]
    alter @mn7[w]=$WT_array[$&index]
    alter @mn6[w]=$WT_array[$&index]

    sp lin 1000 10e6 6e9 1 1
    let noise_f = real(minimum(NF))
    print noise_f >> {out}mixer_noise_figure-test.csv
    set appendwriter

    print Vbrf_test >> {out}Vbrf-test.csv
    print RL_test >> {out}RL-test.csv
    print WN_test >> {out}WN-test.csv
    print WT_test >> {out}WT-test.csv
    let index = index + 1

 end
.endc