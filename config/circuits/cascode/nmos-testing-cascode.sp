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
set w0_array = ( {cascode-w0_array} )
set w1_array = ( {cascode-w1_array} )
set r_array = ( {cascode-r_array} )
set i = {num_samples}
let index = 1
repeat $i
    let r_test = $r_array[$&index]
    alter @mn0[w] = $w0_array[$&index]
    alter @mn1[w] = $w1_array[$&index]
    alter R0 = r_test
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
