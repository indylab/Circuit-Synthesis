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
R6 (net7 0)  50

R0 (vdd Vop)  350
R1 (vdd Von)  350

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

set Vbrf_array = ( {Vbrf_array} )
set RL_array = ( {RL_array} )
set WN_array = ( {WN_array} )
set WT_array = ( {WT_array} )

set i = {num_samples}
let index = 1
repeat $i

    alter V0[dc] = $Vbrf_array[$&index]
    alter @R0[r] = $RL_array[$&index]
    alter @R1[r] = $RL_array[$&index]
    alter @mn1[w] = $WN_array[$&index]
    alter @mn5[w] = $WN_array[$&index]
    alter @mn4[w] = $WN_array[$&index]
    alter @mn2[w] = $WN_array[$&index]
    alter @mn7[w] = $WT_array[$&index]
    alter @mn6[w] = $WT_array[$&index]

    AC lin 1001 5G 6G
    plot mag(v(net5))
    print Vm(net5)
    let magf1= Vm(net5)[775]
    print magf1
    let magf3=Vm(net5)[573]
    print magf3
    let HD3= db(magf1/magf3)
    print HD3 >> {out}mixer_distortion.csv
    set appendwriter
    print $Vbrf_array[$&index] >> {out}Vbrf.csv
    print $RL_array[$&index] >> {out}RL.csv
    print $WN_array[$&index] >> {out}WN.csv
    print $WT_array[$&index] >> {out}WT.csv
    let index = index + 1

end
.endc


