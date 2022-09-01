circuit: LNA
.include {model_path}
.option TEMP=27C

R2 vdd net7 700
R1 net2 net9 10k
R0 vdd net2 800
C1 net7 net6 1p
C0 net8 net9 10p
Lg net9 net4 14.8n
Ld vdd net7 4.4n
Ls net5 0 58.3p
V0 vdd 0 dc 1.2
mn2 net2 net2 0 0 NMOS w=7u l=45n
mn1 net7 vdd net1 0 NMOS w=51u l=45n
mn0 net1 net4 net5 0 NMOS w=51u l=45n
V1 net9 0 dc 0 sin(0 100m 2.4G) portnum 1 z0 50
V2 net6 0 dc 0 ac 0 portnum 2 z0 50

.control

let ls_start = {ls_start}
let ls_test = {ls_start}
let ls_stop = {ls_stop}
let delta_ls = {ls_change}

let ld_start = {ld_start}
let ld_test = {ld_start}
let ld_stop = {ld_stop}
let delta_ld = {ld_change}

let lg_start = {lg_start}
let lg_test = {lg_start}
let lg_stop = {lg_stop}
let delta_lg = {lg_change}

let r_start = {r_start}
let r_test = {r_start}
let r_stop = {r_stop}
let delta_r = {r_change}

let w_start = {w_start}
let w_test = {w_start}
let w_stop = {w_stop}
let delta_w = {w_change}

while ls_test le ls_stop
    alter Ls = ls_test
    while ld_test le ld_stop
        alter Ld = ld_test
        while lg_test le lg_stop
            alter Lg = lg_test
            while r_test le r_stop
                alter R0 = r_test
                while w_test le w_stop
                    alter @mn0[w] = w_test
                    op
                    sp lin 100 1k 100G 1
                    let s21 = abs(S_2_1)
                    let s12 = abs(S_1_2)
                    let s22 = abs(S_2_2)
                    let s11 = abs(S_1_1)
                    let reals11 = real(minimum(S_1_1))

                    let gmax = s21/s12
                    let Gmax = vecmax(gmax)
                    print Gmax >> {out}Gmax.csv

                    let gp = (s21*s21)/(1-s11*s11)
                    let Gp = vecmax(gp)
                    print Gp >> {out}Gp.csv


                    print reals11 >> {out}s11.csv
                    let nf = real(minimum(NF))
                    print nf >> {out}nf.csv
                    set appendwrite

                    print ls_test >> {out}ls.csv
                    print ld_test >> {out}ld.csv
                    print lg_test >> {out}lg.csv
                    print r_test >> {out}r.csv
                    print w_test >> {out}w.csv
                    let w_test = w_test + delta_w
                end
                let r_test = r_test + delta_r
                let w_test = w_start
            end
            let lg_test = lg_test + delta_lg
            let r_test = r_start
        end
        let ld_test = ld_test + delta_ld
        let lg_test = lg_start
    end
    let ls_test = ls_test + delta_ls
    let ld_test = ld_start
end
.endc
.end