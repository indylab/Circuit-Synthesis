nmos: cs amp

.include C:\Spice64\lib\45nm_CS.pm

.option TEMP=27C

V1 in 0 dc 800m ac 1
Vdd net3 0 dc 1.2
R0 net3 out 500
V2 net2 0 dc 1.2
mn0 net1 in 0 0 NMOS W=5u L=45n
mn1 out net2 net1 0 NMOS W=3u L=45n

.control

let w_start = 4u
let w_stop = 7.5u
let delta_w = 0.25u
let w_test = w_start
let r_start = 200
let r_stop = 500
let delta_r = 18.75
let w1_test = w1_start
let w1_start = 7u
let w1_stop = 10u
let delta_w1 = 0.2u
let w1_test = w1_start

while w_test le w_stop
	alter @mn0[w] = w_test
	while r_test le r_stop
		alter R0 = r_test
		while w1_test le w1_stop
			alter @mn1[w] = w1_test
			op
			let gm = @mn0[gm]
			let a0 = gm*r_test
			let id = @mn0[id]
			let pw = id*1.2
			wrdata pw.csv pw
			wrdata a0.csv a0
			ac dec 1000 1G 100G
			let resp = db(v(out)/v(in)) 
			let measurement_point = vecmax (resp) - 3.0
			meas AC upper_3dB WHEN resp = measurement_point 
			print upper_3dB >> bw.csv
			
			
			
			set appendwrite
			let w1_test = w1_test + delta_w1
		end
		let r_test = r_test + delta_r
		let w1_test = w1_start	
	end
	let w_test = w_test + delta_w
	let r_test = r_start	
end

.endc

.end
