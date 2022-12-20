 Circuit: Mixer
.include {model_path}
.option TEMP=27C

mn4 Von VLOP net2 0 NMOS w=10u l=45.0n
mn5 Von VLON net3 0 NMOS w=10u l=45.0n
mn1 Vop VLOP net3 0 NMOS w=10u l=45.0n
mn2 Vop VLON net2 0 NMOS w=10u l=45.0n

mn7 net2 VRFN 0 0 NMOS w=20u l=45.0n
mn6 net3 VRFP 0 0 NMOS w=20u l=45.0n


R0 vdd Vop 350
R1 vdd Von 350
C0 vdd Vop 1p
C2 vdd Von 1p

V7 vdcrf1 0 dc 750m
V8 vdcrf2 0 dc 750m


V0 VRFP vdcrf1 dc 0 SIN (0 1m 5.775G 0 0 0)
V4 VRFN vdcrf2 dc 0 SIN (0 1m 5.775G 0 0 180)

V1 vdd 0 dc 1.2

V3 VLOP 0 dc 0 PULSE (0 1.2 0 1e-13S 1e-13S 9e-11S 1.8e-10S 0)
V2 VLON 0 dc 0 PULSE (0 1.2 0 1e-13S 1e-13S 9e-11S 1.8e-10S 180)


E1 (net7 0 Vop Von) 1

.control
set wn_array = ( {mixer-wn_array} )
set wt_array = ( {mixer-wt_array} )
set rl_array = ( {mixer-rl_array} )
set vbrf_array = ( {mixer-vbrf_array} )
set i = {num_samples}
let index = 1
repeat $i


alter @V7[dc]= $vbrf_array[$&index]
alter @V8[dc]= $vbrf_array[$&index]

alter @R0[r] = $rl_array[$&index]
alter @R1[r] = $rl_array[$&index]

alter @mn1[w]=$wn_array[$&index]
alter @mn5[w]=$wn_array[$&index]
alter @mn4[w]=$wn_array[$&index]
alter @mn2[w]=$wn_array[$&index]



alter @mn7[w]=$wt_array[$&index]
alter @mn6[w]=$wt_array[$&index]

tran 0.5n 20n
meas tran id1 RMS i(V1) from=10ns to=19ns
let Power=id1*1.2
print Power >> {out}/mixer-PowerConsumption.csv

tran 1n 20n
meas tran RFpp PP v(VRFP) from=5nS to=15nS
meas tran IFpp PP v(net7) from=5nS to=15nS
let Swing = IFpp/2
let Conversion_Gain = IFpp/RFpp


print Swing >> {out}/mixer-Swing.csv
print Conversion_Gain >> {out}/mixer-Conversion_Gain.csv

let wn = $wn_array[$&index]
let wt = $wt_array[$&index]
let rl = $rl_array[$&index]
let vbrf = $vbrf_array[$&index]

print wn >> {out}/mixer-wn.csv
print wt >> {out}/mixer-wt.csv
print rl >> {out}/mixer-rl.csv
print vbrf >> {out}/mixer-vbrf.csv

let index = index + 1
end


.endc
.end