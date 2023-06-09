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
set ls_array = ( {LNA-ls_array} )
set ld_array = ( {LNA-ld_array} )
set lg_array = ( {LNA-lg_array} )
set w_array = ( {LNA-w_array} )
set i = {num_samples}
let index = 1
repeat $i
    alter Ls = $ls_array[$&index]
    alter Ld = $ld_array[$&index]
    alter Lg = $lg_array[$&index]
    alter @mn0[w] = $w_array[$&index]

    sp lin 100 1k 100G 1
    let s21 = abs(S_2_1)
    let s12 = abs(S_1_2)
    let s11 = abs(S_1_1)
    let reals11 = real(minimum(S_1_1))



    let gmax = s21/s12
    let Gmax2 = vecmax(gmax)
    print Gmax2 >> {out}/LNA-Gt.csv


    print reals11 >> {out}/LNA-s11.csv
    let nf = real(minimum(NF))
    print nf >> {out}/LNA-nf.csv

    print $ls_array[$&index]
    print $ls_array[$&index] >> {out}/LNA-ls.csv
    print $ld_array[$&index] >> {out}/LNA-ld.csv
    print $lg_array[$&index] >> {out}/LNA-lg.csv
    print $w_array[$&index] >> {out}/LNA-w.csv

    let index = index + 1
end

.endc
.end