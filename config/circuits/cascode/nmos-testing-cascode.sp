Circuit: Cascode Stage Amplifier
.include {model_path}
.option TEMP=27C
.option rshunt = 1.0e12
V1 in 0 dc 625m AC 1
Vdd net3 0 dc 2.5
R0 net3 out 10k
V2 net2 0 dc 1
mn0 net1 in 0 0 NMOS W=10u L=45n
mn1 out net2 net1 0 NMOS W=3u L=45n
.control
set w0_array = ( {cascode-w0_array} )
set w1_array = ( {cascode-w1_array} )
set r_array = ( {cascode-r_array} )
set i = {num_samples}
let index = 1
repeat $i
    alter R0 = $r_array[$&index]
    alter @mn0[w] = $w0_array[$&index]
    alter @mn1[w] = $w1_array[$&index]
	AC dec 100 1MEG 100G
    let resp = maximum(vdb(out))
    let a0 = resp
    let measurement_point = resp - 3.0
    meas AC upper_3dB WHEN vdb(out) = measurement_point
	let bw = upper_3dB
	  let id1 = @mn0[id]
       let pw = id1 * 2.5
    let r = $r_array[$&index]
    let w0 = $w0_array[$&index]
    let w1 = $w1_array[$&index]
    print r >> {out}/cascode-r.csv
    print w0 >> {out}/cascode-w0.csv
    print w1 >> {out}/cascode-w1.csv
    print bw >> {out}/cascode-bw.csv
    print pw >> {out}/cascode-pw.csv
    print a0 >> {out}/cascode-a0.csv
    set appendwrite
    let index = index + 1
end
.endc
.end