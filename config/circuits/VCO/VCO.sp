nmos:VCO
.include {model_path}
.option TEMP=27C
mn6 Vcont vout2 Vcont 0 NMOS w=80u l=45n
mn5 Vcont vout1 Vcont 0 NMOS w=80u l=45n
mn3 net3 net3 0 0 NMOS w=11u l=45n
mn4 net5 net3 0 0 NMOS w=150u l=45n
mn1 vout1 vout2 net5 0 NMOS w=120u l=45n
mn0 vout2 vout1 net5 0 NMOS w=120u l=45n
L1 net4 vout1 4n
L0 net4 vout2 4n
R1 net4 vout2 2200
R0 net4 vout1 2200
C1 net4 vout2 150.0f ic=1
C0 net4 vout1 150.0f
V1 Vcont 0 dc 0.6
V0 net4 0 dc 1.2
I3 net4 net3  dc 300u
E1 (vo 0 vout1 vout2) 1
.control
set wn_array = ( {vco-wn_array} )
set wt_array = ( {vco-wt_array} )
set wv_array = ( {vco-wv_array} )
set lt_array = ( {vco-lt_array} )
set i = {num_samples}
let index = 1
repeat $i
   let f = 3G
   let Q = 30
alter @mn0[w] = $wn_array[$&index]
alter @mn1[w] = $wt_array[$&index]
alter @mn4[w] = $wt_array[$&index]
alter @mn5[w] = $wv_array[$&index]
alter @mn6[w] = $wv_array[$&index]
alter L1 = $lt_array[$&index]
alter L0 = $lt_array[$&index]
alter R0 = ($lt_array[$&index])*2*pi*f*Q
alter R0 = ($lt_array[$&index])*2*pi*f*Q
alter C1 = 1/(185.15*f*f*($lt_array[$&index]))
alter C0 = 1/(185.15*f*f*($lt_array[$&index]))
let w = $wn_array[$&index]
let w1 = $wt_array[$&index]
let w2 = $wv_array[$&index]
let w3 = $lt_array[$&index]
print w >> {out}/vco-wn.csv
print w1 >> {out}/vco-wt.csv
print w2 >> {out}/vco-wv.csv
print w3 >> {out}/vco-lt.csv
let index = index + 1
tran 0.05n 20n
let Vout = v(Vout1)-(Vout2)
meas tran Voutrms RMS Vout from=10ns to=18ns
let Pout = Voutrms*Voutrms/50
print Pout >> {out}/vco-out_power.csv
meas tran Itot RMS i(V0) from=10ns to=18ns
let DC_Power= Itot * 1.2
print DC_Power >> {out}/vco-power_consumption.csv
alter @V1[dc] = 0
tran 0.05n 20n
meas tran tdiff1 TRIG v(vo) VAL=0 cross=50 TARG v(vo) VAL=0 cross=70
let f1 = 10/(tdiff1)
alter @V1[dc] = 1.2
tran 0.05n 20n
meas tran tdiff2 TRIG v(vo) VAL=0 cross=50 TARG v(vo) VAL=0 cross=70
let f2 = 10/(tdiff2)
let TR = abs((tran2.f1)-f2)
print TR >> {out}/vco-tuningrange.csv
end
.endc
.end