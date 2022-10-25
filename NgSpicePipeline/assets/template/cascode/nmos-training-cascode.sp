nmos: cs amp

.include {model_path}

.option TEMP=27C

V1 in 0 dc 800m ac 1
Vdd net3 0 dc 1.2
R0 net3 out 500
V2 net2 0 dc 1.2
mn0 net1 in 0 0 NMOS W=5u L=45n
mn1 out net2 net1 0 NMOS W=3u L=45n

.control

let w_start = {cascode-w0_start}
let w_stop = {cascode-w0_stop}
let delta_w = {cascode-w0_change}
let w_test = w_start
let r_start = {cascode-r_start}
let r_stop = {cascode-r_stop}
let delta_r = {cascode-r_change}
let w1_test = r1_start
let w1_start = {cascode-w1_start}
let w1_stop = {cascode-w1_stop}
let delta_w1 = {cascode-w1_change}
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
			let cgd = @mn1[cgd]
			let cdb = abs(@mn1[cdb])
			let cp = (1+a0)*cgd+cdb
			let bw = 1/(2*pi*r_test*cp)
			print bw cgd cdb
			wrdata {out}/cascode-r.csv r_test
            wrdata {out}/cascode-w0.csv w_test
            wrdata {out}/cascode-w1.csv w1_test
            wrdata {out}/cascode-bw.csv bw
            wrdata {out}/cascode-pw.csv pw
            wrdata {out}/cascode-a0.csv a0
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
