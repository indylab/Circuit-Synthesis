circuit: TwoStageAmplifier
.include {model_path}
.option TEMP=27C

V1 net14 0 dc 1
V2 net10 0 dc 1
V3 net4 0 dc 2.4
V4 net3 0 dc 2.1
V5 net2 0 dc 1.5
V6 net9 0 dc 700m
V7 vdd 0 dc 3.3

mn0 out1 net14 0 0 NMOS W=2.4u L=45n
mn1 out2 net10 0 0 NMOS W=2.4u L=45n

mn9 out4 net2 net11 0 NMOS W=58.5u L=117n
mn6 out3 net2 net6 0 NMOS W=58.5u L=117n

mn8 net11 net13 net8 0 NMOS W=25u L=45n
mn5 net6 net1 net8 0 NMOS  W=25u L=45n

mn7 net8 net9 0 0 NMOS W=52u L=45n

V8 net13 0 dc 1.2 ac 1
V9 net1 0 dc 1.2 ac 1

mp5 out1 out3 vdd vdd PMOS W=6u L=45n
mp4 out2 out4 vdd vdd PMOS W=6u L=45n

mp1 net12 net4 vdd vdd PMOS W=13u L=45n
mp0 net7 net4 vdd vdd PMOS W=13u L=45n

mp3 out4 net3 net12 vdd PMOS W=94u L=180n
mp2 out3 net3 net7 vdd PMOS W=94u L=180n

C4 out3 out1 14p
C3 out4 out2 14p
C1 out2 0 5p
C0 out1 0 5p

.control
set w0_array = ( {ts-w0_array} )
set w1_array = ( {ts-w1_array} )
set w2_array = ( {ts-w2_array} )
set i = {num_samples}
let index = 1
repeat $i
    alter @mn5[w] = $w0_array[$&index]
	alter @mn8[w] = $w0_array[$&index]
	alter @mn7[w] = $w1_array[$&index]
	alter @mp5[w] = $w2_array[$&index]
	alter @mp4[w] = $w2_array[$&index]
    op
    let id2 = @mp4[id]
    let id1 = @mn8[id]
    let pw = (id1+id2)*3.3
    let av = v(out2)/v(net1)
    let w0 = $w0_array[$&index]
    let w1 = $w1_array[$&index]
    let w2 = $w2_array[$&index]
    print w0 >> {out}/ts-w0.csv
    print w1 >> {out}/ts-w1.csv
    print w2 >> {out}/ts-w2.csv
    print pw >> {out}/ts-pw.csv
    print av >> {out}/ts-a0.csv

    ac dec 1000 1G 100G
    let resp = db(v(out2)/v(net1))
    let measurement_point = vecmax (resp) - 3.0
    meas AC upper_3dB WHEN resp = measurement_point
    print upper_3dB >> {out}/ts-bw.csv
    set appendwrite
    let index = index + 1

end
.endc