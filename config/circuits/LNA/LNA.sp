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

let ls_start = {LNA-ls_start}
let ls_test = {LNA-ls_start}
let ls_stop = {LNA-ls_stop}
let delta_ls = {LNA-ls_change}

let ld_start = {LNA-ld_start}
let ld_test = {LNA-ld_start}
let ld_stop = {LNA-ld_stop}
let delta_ld = {LNA-ld_change}

let lg_start = {LNA-lg_start}
let lg_test = {LNA-lg_start}
let lg_stop = {LNA-lg_stop}
let delta_lg = {LNA-lg_change}

let w_start = {LNA-w_start}
let w_test = {LNA-w_start}
let w_stop = {LNA-w_stop}
let delta_w = {LNA-w_change}

while ls_test le ls_stop
    alter Ls = ls_test
    while ld_test le ld_stop
        alter Ld = ld_test
        while lg_test le lg_stop
            alter Lg = lg_test
            while w_test le w_stop
                alter @mn0[w] = w_test
                sp lin 100 1k 100G 1
                let s21 = abs(S_2_1)
                let s12 = abs(S_1_2)
                let s11 = abs(S_1_1)
                let reals11 = real(minimum(S_1_1))

                let gmax = s21/s12
                let Gmax2 = vecmax(gmax)
                print Gmax2 >> {out}/lna-Gp.csv

                print reals11 >> {out}/lna-s11.csv
                let nf = real(minimum(NF))
                print nf >> {out}/lna-nf.csv


                print ls_test >> {out}/LNA-ls.csv
                print ld_test >> {out}/LNA-ld.csv
                print lg_test >> {out}/LNA-lg.csv
                print w_test  >> {out}/LNA-w.csv
                set appendwrite

                let w_test = w_test + delta_w
            end
            let w_test = w_start
            let lg_test = lg_test + delta_lg
        end
        let ld_test = ld_test + delta_ld
        let lg_test = lg_start
    end
    let ls_test = ls_test + delta_ls
    let ld_test = ld_start
end
.endc
.end