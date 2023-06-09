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

let w_start = {ts-w0_start}
let w_stop = {ts-w0_stop}
let delta_w = {ts-w0_change}
let w_test = w_start
let w2_start = {ts-w2_start}
let w2_stop = {ts-w2_stop}
let delta_w2 = {ts-w2_change}
let w2_test = w2_start
let w1_start = {ts-w1_start}
let w1_stop = {ts-w1_stop}
let delta_w1 = {ts-w1_change}
let w1_test = w1_start

while w_test le w_stop
	alter @mn5[w] = w_test
	alter @mn8[w] = w_test
	while w2_test le w2_stop
		alter @mp5[w]= w2_test
		alter @mp4[w]= w2_test
		while w1_test le w1_stop
		    alter @mn7[w]= w1_test
			op 
			let id2 = @mp4[id]
			let id1 = @mn8[id]
			let pw = (id1+id2)*3.3
			let av = v(out2)/v(net1)
			wrdata {out}/ts-w0.csv w_test
			wrdata {out}/ts-w1.csv w1_test
			wrdata {out}/ts-w2.csv w2_test
			wrdata {out}/ts-pw.csv pw
			wrdata {out}/ts-a0.csv av

			ac dec 1000 1G 100G
			let resp = db(v(out2)/v(net1)) 
			let measurement_point = vecmax (resp) - 3.0
			meas AC upper_3dB WHEN resp = measurement_point 
			print upper_3dB >> {out}/ts-bw.csv
			
			set appendwrite
			let w1_test = w1_test + delta_w1
		end
		let w2_test = w2_test + delta_w2
		let w1_test = w1_start
	end
	let w_test = w_test + delta_w
	let w2_test = w2_start
end
.endc